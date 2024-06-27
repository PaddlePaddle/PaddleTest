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
    class PrimitiveOp_1bf525165d16c2a153a298dc0916bf81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6b58699b49ff9231455c42b6ed66576(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.33181232213974]], [[0.06974681466817856]], [[0.4563100039958954]], [[0.40709802508354187]], [[0.10137591511011124]], [[0.21714185178279877]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7579535841941833]], [[0.5940682888031006]], [[0.5430742502212524]], [[0.573845624923706]], [[0.7277628183364868]], [[0.6181967258453369]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_e6c5ec80c69cd71e72a5fdc8c7c399e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.17835448682308197]], [[0.4390904903411865]], [[0.30827003717422485]], [[0.45996543765068054]], [[0.18206551671028137]], [[0.3943363130092621]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6730959415435791]], [[0.6264330148696899]], [[0.724555253982544]], [[0.5223374366760254]], [[0.7338444590568542]], [[0.8147628903388977]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_36ee1ec77ca5536bab638d289c99803e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2cf34139b4eef42412b3115d46f69e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5627d7089bc7ccf837f32569031cb2da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9f3a79b85284653a72bc1c586f62c13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12096, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad10572b5afc38c39c8ccb08f32ee708(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f3a79b85284653a72bc1c586f62c13
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_787f053952c37fbaeda5e3810907422b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2e421643d1e9a7221f991f14614920b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22ed387fdf4b1a5791d0d375a6917ea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.11059033870697021, 0.07338421791791916]], [[0.12060432136058807, 0.29075783491134644]], [[0.2402641475200653, 0.3665030002593994]], [[0.07487303018569946, 0.09319182485342026]], [[0.00415261322632432, 0.4588662385940552]], [[0.37556031346321106, 0.12091569602489471]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.1949375867843628, 0.19650663435459137]], [[0.029328227043151855, 0.49953198432922363]], [[0.19212405383586884, 0.4960818290710449]], [[0.413517564535141, 0.10393795371055603]], [[0.4010055661201477, 0.46761366724967957]], [[0.35190922021865845, 0.22220514714717865]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_52f01ef85ccad259c6bdfd98b5735ebf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.043390195816755295, 0.022145669907331467]], [[0.1635356843471527, 0.41414403915405273]], [[0.09432101249694824, 0.048812177032232285]], [[0.11243130266666412, 0.03282276540994644]], [[0.34049803018569946, 0.03584573045372963]], [[0.23450057208538055, 0.37127581238746643]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.1949375867843628, 0.19650663435459137]], [[0.029328227043151855, 0.49953198432922363]], [[0.19212405383586884, 0.4960818290710449]], [[0.413517564535141, 0.10393795371055603]], [[0.4010055661201477, 0.46761366724967957]], [[0.35190922021865845, 0.22220514714717865]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    
    class PrimitiveOp_6ae748e3ee78da0b91c7817413d3dc0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1aab1a88f19a6c240e52861735fbb94e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ae748e3ee78da0b91c7817413d3dc0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.48046234250068665, 0.22905594110488892]], [[0.2770739495754242, 0.3431546986103058]], [[0.24031347036361694, 0.48387277126312256]], [[0.4746449291706085, 0.4006763994693756]], [[0.10624674707651138, 0.05708790943026543]], [[0.49964481592178345, 0.2771390378475189]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cd4ad177bf59e7b8182006c518bdcee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.2276126593351364, 0.34858253598213196, 0.47927117347717285, 0.4129710793495178, 0.017648370936512947, 0.38352295756340027, 0.190532848238945, 0.46824896335601807, 0.18940813839435577, 0.2642250657081604, 0.2662023603916168, 0.47233644127845764, 0.2409619688987732, 0.2888906002044678, 0.39573538303375244, 0.3080037832260132], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_56553316d0dbddf51670765857858e50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2276126593351364, 0.34858253598213196, 0.47927117347717285, 0.4129710793495178, 0.017648370936512947, 0.38352295756340027, 0.190532848238945, 0.46824896335601807, 0.18940813839435577, 0.2642250657081604, 0.2662023603916168, 0.47233644127845764, 0.2409619688987732, 0.2888906002044678, 0.39573538303375244, 0.3080037832260132], dtype='float32').reshape([16]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1d8785fab36e42b156399b6fd6430c17(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300], dtype='float32'),
                paddle.static.InputSpec(shape=[300], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76c5cb57b95ec8abfdf740ca8bd2bfcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d8785fab36e42b156399b6fd6430c17
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76c5cb57b95ec8abfdf740ca8bd2bfcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d8785fab36e42b156399b6fd6430c17
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63299c2e4d225192c2b0be218c743f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c54fbf14b292b0ea5224df92673802b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c4adb159a881a57198709244f578ad58(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d47836044ada441c30bbd1bb43c398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f4deaf6fd6d801a876df9c234a227795(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ee1c02db88ba5f2372a7802d533bba5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9bc4636b3ea34ab0e9db2ad06ef24173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c54fbf14b292b0ea5224df92673802b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0b7f661e3bf6a7a4daf8a03c5266065(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25037649273872375, 0.04901019111275673, 0.029957201331853867, 0.3703627288341522], [0.4998129904270172, 0.20214392244815826, 0.4599210023880005, 0.23039565980434418], [0.4898363947868347, 0.01842341385781765, 0.3382553458213806, 0.48959681391716003], [0.2272232323884964, 0.34871217608451843, 0.3543879985809326, 0.38190561532974243], [0.29566892981529236, 0.25489845871925354, 0.12884438037872314, 0.22511659562587738]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.23842979967594147, 0.2781031131744385, 0.1888388842344284, 0.38554295897483826], [0.26523661613464355, 0.34089452028274536, 0.22515225410461426, 0.19126184284687042], [0.1216023787856102, 0.2251485288143158, 0.3785667419433594, 0.4806677997112274], [0.4923000931739807, 0.4137409031391144, 0.027203183621168137, 0.44860342144966125], [0.38564565777778625, 0.14669828116893768, 0.18097509443759918, 0.29368749260902405]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_177a4e91d094a30cb01a43da75d1e858(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 5376, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34014d3d3c79334f9a3b341aa65ab537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_177a4e91d094a30cb01a43da75d1e858
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b245b1bfb03b3cebd65892f188f4363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20757071673870087, 0.005273854359984398, 0.16517065465450287, 0.18617504835128784], [0.3260256052017212, 0.20955143868923187, 0.40184587240219116, 0.18627451360225677], [0.0242769755423069, 0.11374565213918686, 0.2671873867511749, 0.18605898320674896], [0.3260256052017212, 0.20955143868923187, 0.40184587240219116, 0.18627451360225677], [0.0242769755423069, 0.11374565213918686, 0.2671873867511749, 0.18605898320674896]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.4676962196826935, 0.28657740354537964, 0.4662608802318573, 0.30012109875679016], [0.1984112560749054, 0.3507075607776642, 0.11108946800231934, 0.1902817189693451], [0.4010750949382782, 0.1335587352514267, 0.35621824860572815, 0.25092580914497375], [0.1984112560749054, 0.3507075607776642, 0.11108946800231934, 0.1902817189693451], [0.4010750949382782, 0.1335587352514267, 0.35621824860572815, 0.25092580914497375]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2fa0793049a656443ce9fe67b9e7eff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.32824042439460754], [0.053689487278461456], [0.04877207800745964], [0.24639740586280823], [0.06132432818412781], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.33700039982795715], [0.1544954627752304], [0.3554002642631531], [0.4550243616104126], [0.38996535539627075], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_de7aa4191288abc9f79ec0b5d54134a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.23870912194252014], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.3334523141384125], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2dad2ddf13ef0849e8a5b569a71122a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.32824042439460754], [0.11407893896102905], [0.04877207800745964], [0.24639740586280823], [0.4660728871822357], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.33700039982795715], [0.13599033653736115], [0.16089512407779694], [0.4550243616104126], [0.2522077262401581], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_97de8052ca91eda5001ce1ba5ebf7798(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.477491170167923], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.2988268733024597], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_980565ffac6c1676132c3d4894001908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.3819327652454376], [0.053689487278461456], [0.17333781719207764], [0.3545892834663391], [0.06132432818412781], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.1544954627752304], [0.3554002642631531], [0.06358063966035843], [0.38996535539627075], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9c8877c82efefe2bdf6d2f6db35137e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.23870912194252014], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.3334523141384125], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_77f5a15eef1222a60aa9b608abcf1b8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08513391762971878], [-0.0048834290355443954], [0.09082716703414917], [0.06779509782791138], [-0.02920367568731308], [0.020875904709100723], [0.1249118521809578], [-0.005090379621833563], [0.07913517206907272]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.02909252792596817], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2906250128bacd6817ee26723e5dac7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.3819327652454376], [0.11407893896102905], [0.17333781719207764], [0.3545892834663391], [0.4660728871822357], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.13599033653736115], [0.16089512407779694], [0.06358063966035843], [0.2522077262401581], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2c03fe92aa0efedc9d68d3c3cd6ccf27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.477491170167923], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.2988268733024597], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e7a553faeca3b83810d40ced3a408810(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.034133654087781906], [0.0038358664605766535], [0.06032535061240196], [0.04406944662332535], [-0.018659166991710663], [-0.029324453324079514], [0.09581932425498962], [0.04879090562462807], [0.0004284780006855726]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.08513391762971878], [-0.0048834290355443954], [0.09082716703414917], [0.06779509782791138], [-0.02920367568731308], [0.020875904709100723], [0.09581932425498962], [-0.005090379621833563], [0.07913517206907272]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_a6efec15ee9daa5333b107372c82822c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0], [-0.0], [0.0], [0.30361858010292053], [-0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[3.4941341876983643], [2.2730965614318848], [-0.5056218504905701], [-0.5383695960044861], [-0.5651114583015442], [1.7118940353393555], [0.0], [1.1043304204940796], [-183.68899536132812]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_1af58d553c5e7b6ec645021f13cf8a4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2b1e6bb09aba51197669f46cd42c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.0004822202608920634]], [[0.40181925892829895]], [[0.4493107199668884]], [[0.1114136129617691]], [[0.1425887644290924]], [[0.28795698285102844]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.585241973400116]], [[0.7111523747444153]], [[0.7787365317344666]], [[0.7846488356590271]], [[0.5292868614196777]], [[0.5829117894172668]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_9c0cbb1777c11c3b5772b34b251a7fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.29122838377952576]], [[0.4045177400112152]], [[0.12232992798089981]], [[0.43076565861701965]], [[0.004029393196105957]], [[0.3265424966812134]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.5337342023849487]], [[0.759053647518158]], [[0.5657528042793274]], [[0.7224377989768982]], [[0.5274229049682617]], [[0.5056599378585815]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_6a2712d0e38e262dd5356c4fc102e730(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8521c16572cda0958a5430dd057a80cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a2712d0e38e262dd5356c4fc102e730
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88173d5c5ea1f84b122a32a28c3a7e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e1919e04593ec42eda1962b2c9b263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_058d7920382cf156edd90d10cbccd897(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fb7e4bbce32cef9350027ea3a56b438(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88173d5c5ea1f84b122a32a28c3a7e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ddef26f2e6f7fd18ffec68c34f21fc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42444175481796265, 0.42435893416404724, 0.32613885402679443, 0.17775212228298187], [0.4992370307445526, 0.014581345021724701, 0.47447216510772705, 0.19020026922225952], [0.03811301290988922, 0.19911915063858032, 0.02669600583612919, 0.4305139482021332], [0.4992370307445526, 0.014581345021724701, 0.47447216510772705, 0.19020026922225952], [0.03811301290988922, 0.19911915063858032, 0.02669600583612919, 0.4305139482021332], [0.3735699951648712, 0.3213905096054077, 0.09645315259695053, 0.05379442125558853], [0.3735699951648712, 0.3213905096054077, 0.09645315259695053, 0.05379442125558853]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.09077489376068115, 0.03478417918086052, 0.29913225769996643, 0.35426220297813416], [0.2355150431394577, 0.04002285748720169, 0.34607771039009094, 0.32934725284576416], [0.08519759029150009, 0.19337064027786255, 0.40454861521720886, 0.26980942487716675], [0.2355150431394577, 0.04002285748720169, 0.34607771039009094, 0.32934725284576416], [0.08519759029150009, 0.19337064027786255, 0.40454861521720886, 0.26980942487716675], [0.3071763813495636, 0.03489753603935242, 0.35671326518058777, 0.4256373643875122], [0.3071763813495636, 0.03489753603935242, 0.35671326518058777, 0.4256373643875122]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_cd4c7f79661bf690cbcf40118f518a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd4c7f79661bf690cbcf40118f518a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_224fddb1ca6622431af877bc9b3df646(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7098e869ec6bbcc523f1f9ce070f283(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b51bdd5f5a7dc1d7ec0f4402494c9f8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.04365863651037216, 0.14801615476608276, 0.059532683342695236, 0.1973104625940323, 0.10881417244672775], dtype='float32').reshape([6]),
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_54c874c04ebd91bbc75dc45a2a38f90e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.21491502225399017, 0.4360129237174988, 0.2119390070438385, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_537bb4e866226c3d115e2b89246e1751(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45263877511024475, 0.18129011988639832, 0.45884010195732117, 0.49139586091041565, 0.3470446765422821, 0.2975061535835266], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.20852655172348022, 0.4761844575405121, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b47630c56f9692ccf5222b02c91f0729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.35412517189979553, 0.2729951739311218, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.06498825550079346, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_72dab4f028cf1de1fe4ba35081607118(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.18129011988639832, 0.4002115726470947, 0.3979283273220062, 0.3470446765422821, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.4002115726470947, 0.4761844575405121, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_7e8b29fac1c23cf97dc7fd0e7df2f595(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.35412517189979553, 0.2729951739311218, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.43007832765579224, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_821772acec016f3984fc5b0f81ed8899(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_69e07130dc0b1fa7a2ac6cb4124561bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.4360129237174988, 0.43007832765579224, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e47dec68c91fa93ccd8bac8f72dbbf90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.07526008784770966, 0.02078755758702755, -0.02064768597483635, 0.003164077177643776, 0.009952809661626816, 0.030713750049471855], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0, 0.0, -0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_fda384f0a99189c95d15b8a09d1851ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1936129629611969, 0.2076808512210846, 0.27411386370658875, 0.22873049974441528, 0.3200559914112091, 0.1432904452085495], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4512457847595215, 0.33517807722091675, 0.3336833119392395, 0.48379015922546387, 0.37571316957473755, 0.20266865193843842], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_fcaaaa3c0546e3c542e9319b9ed44bfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.17710404098033905, 0.3038899004459381, 0.4278310537338257, 0.32100868225097656, 0.2117002010345459, 0.24121758341789246], dtype='float32').reshape([6]),
                paddle.to_tensor([0.26647061109542847, 0.4544703960418701, 0.3953688144683838, 0.16899171471595764, 0.40512338280677795, 0.41586926579475403], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_64f870eef914bed48dce1f0099518e98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45263877511024475, 0.37170305848121643, 0.45884010195732117, 0.49139586091041565, 0.44280150532722473, 0.2975061535835266], dtype='float32').reshape([6]),
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.20852655172348022, 0.3979283273220062, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_6b2f50eaee499f1cfbfe527db2b4abd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.4360129237174988, 0.43007832765579224, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.06498825550079346, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_7f7f67b6e90b7286d3fb0ad84e7e1353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.01364241074770689, 1.354771375656128, -1.252467393875122, 0.07299936562776566, 0.31902867555618286, 0.864149808883667], dtype='float32').reshape([6]),
                paddle.to_tensor([0.7114414572715759, 1.0737632513046265, -1.5060021877288818, 0.9982069134712219, -0.8928132057189941, -0.3794630169868469], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_fffd7995680d3cc9fdf22434d03b8290(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6efa6937ca8ef96bbf0e57bc40d4d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee1c02db88ba5f2372a7802d533bba5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bc4636b3ea34ab0e9db2ad06ef24173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fffd7995680d3cc9fdf22434d03b8290(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_32804f7508f85e1a6c9d50ff62f677cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d8ab50c79c4b22c03166d8ef543b0534(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32804f7508f85e1a6c9d50ff62f677cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84fe50643aaf98f94858ab0985e2df5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.2594594955444336, 0.22706426680088043, 0.28697821497917175, 0.16908468306064606, 0.16174711287021637, 0.1938442438840866, 0.42815297842025757, 0.11389681696891785, 0.02597997337579727, 0.2570507228374481, 0.23708491027355194, 0.05830814689397812, 0.4899848997592926, 0.2215232402086258, 0.23221857845783234, 0.024046774953603745, 0.4813443124294281, 0.1871427595615387, 0.23423327505588531, 0.1138496994972229, 0.25865429639816284, 0.3344466984272003, 0.28236815333366394, 0.04511779919266701], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_1a641cb51fb3b8dcefc5f61669a075cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2594594955444336, 0.22706426680088043, 0.28697821497917175, 0.16908468306064606, 0.16174711287021637, 0.1938442438840866, 0.42815297842025757, 0.11389681696891785, 0.02597997337579727, 0.2570507228374481, 0.23708491027355194, 0.05830814689397812, 0.4899848997592926, 0.2215232402086258, 0.23221857845783234, 0.024046774953603745, 0.4813443124294281, 0.1871427595615387, 0.23423327505588531, 0.1138496994972229, 0.25865429639816284, 0.3344466984272003, 0.28236815333366394, 0.04511779919266701], dtype='float32').reshape([24]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68116d520314dfe4e32060cebcab2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8b7e5638ae41e77d63b42c0642fb861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aceac5bc2bf0c9efd501145333912648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4a362d86b9a93da5d962b9491cc8382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68116d520314dfe4e32060cebcab2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d678f25cbdc7c0c3059d5677c915c981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.08582065254449844, 0.4129891097545624, 0.49300137162208557, 0.21711167693138123], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_bf2ff873c46be6c59732aafac24a96f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.08582065254449844, 0.4129891097545624, 0.49300137162208557, 0.21711167693138123], dtype='float32').reshape([4]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
            ]


    
    class PrimitiveOp_f090e4ebe749ea2cde6f0c3d397c7d0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_633ca9186c88ae9ce23a470677ade55e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f090e4ebe749ea2cde6f0c3d397c7d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a9cf8bc37b9ebba6a0bdb54231ae6abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f090e4ebe749ea2cde6f0c3d397c7d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40899add8920b8687e8df0ac80468691(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21024103462696075, 0.146906316280365, 0.3382836878299713, 0.3186817169189453], [0.30262491106987, 0.07154060900211334, 0.4640345871448517, 0.44525519013404846], [0.3060167729854584, 0.30472156405448914, 0.3321487009525299, 0.4837305247783661], [0.02388036996126175, 0.12797322869300842, 0.10076813399791718, 0.4382765293121338], [0.02388036996126175, 0.12797322869300842, 0.10076813399791718, 0.4382765293121338], [0.3060167729854584, 0.30472156405448914, 0.3321487009525299, 0.4837305247783661]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.36117079854011536, 0.3021794855594635, 0.3944045305252075, 0.16512134671211243], [0.37474554777145386, 0.4685516059398651, 0.3396102488040924, 0.0575605146586895], [0.10093643516302109, 0.15885069966316223, 0.27939414978027344, 0.3892151415348053], [0.15035083889961243, 0.31797781586647034, 0.42872291803359985, 0.4998945891857147], [0.15035083889961243, 0.31797781586647034, 0.42872291803359985, 0.4998945891857147], [0.10093643516302109, 0.15885069966316223, 0.27939414978027344, 0.3892151415348053]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_b59aaf6531a893d26acef318cd7a62fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4699752926826477, 0.20170211791992188, 0.33375540375709534, 0.3653546869754791], [0.09326086193323135, 0.3276481330394745, 0.1447921097278595, 0.30204591155052185], [0.2558746635913849, 0.3609501123428345, 0.29354995489120483, 0.09777697175741196], [0.23382103443145752, 0.29737725853919983, 0.19029706716537476, 0.21787863969802856], [0.4699752926826477, 0.20170211791992188, 0.33375540375709534, 0.3653546869754791]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.15726704895496368, 0.25012800097465515, 0.49190860986709595, 0.3168145418167114], [0.4363638758659363, 0.26189401745796204, 0.2109360694885254, 0.028239434584975243], [0.3192857503890991, 0.24466922879219055, 0.1895640790462494, 0.1258022040128708], [0.4110543727874756, 0.040166862308979034, 0.39998859167099, 0.031311191618442535], [0.15726704895496368, 0.25012800097465515, 0.49190860986709595, 0.3168145418167114]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_73ece6d168a08bd3fa887a7c95aa72af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0434f312aa909d37a195006a970b602(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ae128115352692b1cbb0e8b72045fad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c0434f312aa909d37a195006a970b602(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_79f7db25226187c456fb995314d4eeb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_7ef18f9b7952128ac492f0ff155b148b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_cc62686e1541e14d7ddcb163e456a83e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_f35281470df6c37a86839b15b0749a1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0809708684682846]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_7ef18f9b7952128ac492f0ff155b148b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c36dbf5612da382e55aba14835cfc40d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2fb1fbcce67d5f4f396b7da1b5f4f606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08737017959356308]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0809708684682846]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_e56a2a1ecb05cba4e6844f0307db8c1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0732436552643776]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ff1cb0d4a1e61abdb92aab88c95d533e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.2049286961555481], [0.05052289366722107], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.37558823823928833], [0.3696064352989197], [0.3602122664451599], [0.49161165952682495], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_548cfe56a14f6230797aff47d6adc21c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.4184499979019165], [0.4230078458786011], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_003c74cc17546b9ddd7948d0449ec6c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.2049286961555481], [0.05052289366722107], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06345830112695694], [0.37558823823928833], [0.1739027202129364], [0.027879230678081512], [0.49161165952682495], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_7bf502907ff01f454533bd926d6c287e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.13699816167354584], [0.30248284339904785], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ea1c3a6bfd4fa7e120287b86741eb963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.38781505823135376], [0.12123280763626099], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.19266913831233978], [0.3696064352989197], [0.3602122664451599], [0.343851238489151], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e7e93a1e7bdcda73fe8cd23fbacd925a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.4184499979019165], [0.4230078458786011], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ba73f40234619a2da2c9b3cca78bdc2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.009171975776553154], [-0.00045348587445914745], [-0.13099893927574158], [-0.018913721665740013], [-0.02232457511126995], [0.0065928734838962555]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_03ba760056d0dfe25390c0717d3c4237(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.38781505823135376], [0.12123280763626099], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06345830112695694], [0.19266913831233978], [0.1739027202129364], [0.027879230678081512], [0.343851238489151], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_5fb5a9bc112bf56674e0711bf87651c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.13699816167354584], [0.30248284339904785], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_0f464a7a2859d60b2ed207511e7c1e9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.017743036150932312], [0.033636245876550674], [0.046311333775520325], [-0.0053473603911697865], [-0.02140507660806179], [-0.004702110309153795]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.009171975776553154], [-0.0004534857871476561], [-0.13099893927574158], [-0.018913721665740013], [-0.02232457511126995], [0.0065928734838962555]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_976519ccf761e9799e557d922ff3d2c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[1.516933798789978], [1.0134820938110352], [3.82865834236145], [-2.537020206451416], [-0.04295703023672104], [2.402109384536743]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_56a60b59e3f52029d0df6606412d59de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29697567224502563, 0.033938176929950714, 0.12153984606266022, 0.16929808259010315], [0.4059440791606903, 0.4174965023994446, 0.09141463786363602, 0.08322188258171082], [0.15150173008441925, 0.09857954829931259, 0.04490986093878746, 0.2046736776828766], [0.31510642170906067, 0.3103889226913452, 0.10195562988519669, 0.23645362257957458]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.43735429644584656, 0.3411722481250763, 0.35462242364883423, 0.3983362317085266], [0.26059871912002563, 0.4778519570827484, 0.48979929089546204, 0.27005699276924133], [0.19344715774059296, 0.34565991163253784, 0.26136934757232666, 0.3304290473461151], [0.31872597336769104, 0.1560906022787094, 0.2340485155582428, 0.13768287003040314]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd4ad40b9e6cb06109979ddd755a24b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5613305b3a09c8cd302457a7e53c1052(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c13ecf957e906faf67b2a1c0563cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bbf7c0afdbea438b74c5eae4bcac9b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f64cb6fd9abf54063d21adfc7894a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5613305b3a09c8cd302457a7e53c1052(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157d5371e1c02a2020f5d0d1c79c9c7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18468241393566132, 0.33149173855781555, 0.24620825052261353, 0.16647128760814667], [0.18468241393566132, 0.33149173855781555, 0.24620825052261353, 0.16647128760814667], [0.4424126446247101, 0.48034167289733887, 0.41602855920791626, 0.4828370213508606], [0.37814199924468994, 0.462110310792923, 0.3353201150894165, 0.21955512464046478], [0.19167256355285645, 0.028036119416356087, 0.43969249725341797, 0.18763285875320435], [0.1682383418083191, 0.4110064208507538, 0.4892330467700958, 0.2316710650920868], [0.26976892352104187, 0.2744980752468109, 0.3080321252346039, 0.047989457845687866]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.35786253213882446, 0.32739314436912537, 0.3534325063228607, 0.4124147593975067], [0.35786253213882446, 0.32739314436912537, 0.3534325063228607, 0.4124147593975067], [0.16995151340961456, 0.3392655551433563, 0.06221455708146095, 0.45378729701042175], [0.33701011538505554, 0.21520060300827026, 0.3914770781993866, 0.127670019865036], [0.2887539267539978, 0.44512802362442017, 0.24220089614391327, 0.03608894720673561], [0.19074024260044098, 0.1567150354385376, 0.47922074794769287, 0.05587359517812729], [0.2514936625957489, 0.15642105042934418, 0.3574431538581848, 0.026684166863560677]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c089057d08b7de722c2e54697878df9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62fdc432194f6e5a15cd550492016cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a2712d0e38e262dd5356c4fc102e730
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1d7720670a34d59ef5a3952cbaf7d3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638e3651770b5edf27ea99323e5c7485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8417cbf4ddbe166177b4e5bea59889fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48e9a5db03edbdd8a81a2d8d42dd6bc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1d7720670a34d59ef5a3952cbaf7d3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45256844d7c2650d8b266b2670bbca38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee2e600e219759e1ff19d812fdc11c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77fee8356a1b0de98ec213ed503fdeb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bc0c3dc8045c74802061533aeb59dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45256844d7c2650d8b266b2670bbca38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a89dfe01d48a19b39a14b931a6f22cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_573195c405a55b5a52c0ff671e9d5e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09556698054075241, 0.062248557806015015, 0.48012056946754456, 0.1216694638133049], [0.4499690532684326, 0.3386240601539612, 0.13975663483142853, 0.35235655307769775], [0.4499690532684326, 0.3386240601539612, 0.13975663483142853, 0.35235655307769775], [0.3378070592880249, 0.048087358474731445, 0.4574849605560303, 0.04036188870668411], [0.17514236271381378, 0.24521946907043457, 0.13926689326763153, 0.0350252240896225], [0.1317417472600937, 0.24890102446079254, 0.07940025627613068, 0.47396957874298096]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.1827584058046341, 0.46928292512893677, 0.3330903947353363, 0.32664167881011963], [0.23554666340351105, 0.14961957931518555, 0.36159104108810425, 0.34157228469848633], [0.23554666340351105, 0.14961957931518555, 0.36159104108810425, 0.34157228469848633], [0.133694127202034, 0.19349321722984314, 0.06283392757177353, 0.011135056614875793], [0.2018437683582306, 0.08454585075378418, 0.3254218101501465, 0.44437113404273987], [0.46540915966033936, 0.167648583650589, 0.3530118465423584, 0.1341691017150879]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d54b4c0274255a4200a257626b103198(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f62c64312ccb8d87c12c185fd5e515e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 1, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f90e04327aad6de4461e210516f8bfa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f62c64312ccb8d87c12c185fd5e515e
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1.3719160556793213, 0.5977555513381958, 0.5020655393600464, 0.020979739725589752], [0.5314668416976929, 1.1458791494369507, 0.8261379599571228, 0.6206862330436707]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0bbadf53b0a9033972be39082fed00c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6069, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95ae05c98f4422e854db6739889388cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bbadf53b0a9033972be39082fed00c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_321485876b504aa24b6a10287956155c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3db6dde5bf3f37813420aec60ad447b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 1, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0154d2883246abd51409f884de65812(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3db6dde5bf3f37813420aec60ad447b6
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3960612714290619, 0.04548724740743637, 0.1888345181941986, 10.237210273742676], [0.29948604106903076, 3.9570305347442627, 12.784720420837402, 6.013519287109375]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_a3b7dfba983dcbb80ed2682e3741967a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.08265111595392227], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b8af055c6563e68c27a2cb3674ea1f1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.03426346927881241], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_057d9c8d63faa8ba2459d2aa3a8423e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.08265111595392227], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b34af14cfc4f9e102e017a8eaf61d8f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.3689388930797577], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2bf2e7effa2650c4498074603be12a14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.1922953873872757], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_44d7271673afe3b63d738e1f67999f3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.03426346927881241], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_869014e54dce68fc00b3d66f920dbda9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015536606311798096], [-0.028670985251665115], [0.08992450684309006], [-0.0018401052802801132], [0.001199607620947063]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_003ba39979bea7bf26a9ebb77ec67a30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.1922953873872757], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e5b3b6361b8952548e91debe24441ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.3689388930797577], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_91b25ca2604d1f0a2368cb8a6bc783bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.024802792817354202], [0.09054671227931976], [0.1189025491476059], [-0.006836474873125553], [0.0045682224445044994]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.015536606311798096], [-0.028670985251665115], [0.08992450684309006], [-0.0018401051638647914], [0.0011996077373623848]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c6669f96152f8a3c4fbd06ba0146d7e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[1.6264054775238037], [1.3166429996490479], [0.24371254444122314], [0.7308400273323059], [0.7374016642570496]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_327506a3e0fa3b1dd7c6da9932d96b80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a2712d0e38e262dd5356c4fc102e730
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e56e4f522d59f318883e9e6b7986e56b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6a2aaa8232b44d4b7b9c41761a5acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a2cbc4a6e4fd6a507c47523b68f2aa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd02780250d0b27ca48abce8f252bf24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e56e4f522d59f318883e9e6b7986e56b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12b995b51eca873b436c247f68e9a0a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a33aa867b3964d0aa68dbbb52140d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca761e7333f233d8f0e4e90a9557a97f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4225640db7d7743b5c47ae2e609b9362(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12b995b51eca873b436c247f68e9a0a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fc39a3efa7990a02e240b156aaab3bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53ef35974bf6d2987081095faaed1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b00f4c34395a6755ddb8c2956603745(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54c2eefbb3f8526e1cdbfdb990ee43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fc39a3efa7990a02e240b156aaab3bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62c87acbb50f729504ed4af6fbcef4a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a2712d0e38e262dd5356c4fc102e730
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b01d14df83ac09763ab30092653478d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f9e878396c64b78f0adbf868a6b069f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_573576e0724f739f0d0dc8406be8ee96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e878396c64b78f0adbf868a6b069f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db1bdf21fe9070524e3637f533cdfb60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.30421602725982666, 0.054143913090229034, 0.394016832113266, 0.46182653307914734, 0.3547346591949463, 0.1406528353691101, 0.37224218249320984, 0.43492934107780457, 0.23447169363498688, 0.12492112815380096, 0.23746684193611145, 0.47617754340171814, 0.04644746333360672, 0.10742539912462234, 0.4697968661785126, 0.29846274852752686, 0.3385973274707794, 0.021797627210617065, 0.2388172298669815, 0.046391844749450684], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_3ab5dc5768e0c7e68c916c9ee4ab131c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.30421602725982666, 0.054143913090229034, 0.394016832113266, 0.46182653307914734, 0.3547346591949463, 0.1406528353691101, 0.37224218249320984, 0.43492934107780457, 0.23447169363498688, 0.12492112815380096, 0.23746684193611145, 0.47617754340171814, 0.04644746333360672, 0.10742539912462234, 0.4697968661785126, 0.29846274852752686, 0.3385973274707794, 0.021797627210617065, 0.2388172298669815, 0.046391844749450684], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_ae0f530b28ef9cba57911dd9973099f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.2982136011123657], [0.18588005006313324], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41876405477523804], [0.462017297744751], [0.11372362822294235], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_fd1e882062072551b01464359032bb0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33433759212493896], [0.025396505370736122], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_91a0f4863c2fd241ccc5991e70328361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.43085747957229614], [0.18588005006313324], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41876405477523804], [0.0375184640288353], [0.09816955029964447], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_33b098216120e988fce1cfa49a08c40a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.025396505370736122], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7268e014e110161e11148b2812106119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4131563901901245], [0.2982136011123657], [0.22558467090129852], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.462017297744751], [0.11372362822294235], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_eaf912e4dd458073f7b3a70dc35a1974(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33433759212493896], [0.12095697224140167], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_ce9ee6b82614ae963ed24390e3517704(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06323745846748352], [-0.025388313457369804], [0.017503943294286728], [0.007693326100707054]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_de6faedb6066058e93ae514cf20ce6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4131563901901245], [0.43085747957229614], [0.22558467090129852], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.0375184640288353], [0.09816955029964447], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3d37e6e035d79d496e11bf1e91cff7f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.12095697224140167], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3caf9a74e26c244968642218b2091139(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.026258420199155807], [0.011381367221474648], [0.030490899458527565], [-0.0003965869836974889]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.06323745846748352], [-0.025388313457369804], [0.017503943294286728], [0.007693326100707054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_dd831fc68a90a7ae09ce4202d408e2d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[3.408273696899414], [3.2306909561157227], [0.42592892050743103], [20.398836135864258]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f866141f6138d4c2131ae7ec085de7de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cafd0fbc33e575fd5fbb5139ffc2606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1492fac3a6a451c326968b0b36c1583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bbf7c0afdbea438b74c5eae4bcac9b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f64cb6fd9abf54063d21adfc7894a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cafd0fbc33e575fd5fbb5139ffc2606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_573576e0724f739f0d0dc8406be8ee96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9e878396c64b78f0adbf868a6b069f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3da7dfdbe60188bab47ede9aa830f09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a2712d0e38e262dd5356c4fc102e730
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9bfc4b03f03520a51d9a2b7660ffabf1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6804, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b43cbc98c94fd917839ad0b2c9ee192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9bfc4b03f03520a51d9a2b7660ffabf1
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_430aa3aa9e88a6965d5a1616f124695e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.059797223657369614, 0.3857690691947937, 0.4381629526615143, 0.1027114987373352], [0.4444314241409302, 0.43357622623443604, 0.13710230588912964, 0.45452260971069336], [0.24993541836738586, 0.26625725626945496, 0.2578131854534149, 0.3478696346282959], [0.24993541836738586, 0.26625725626945496, 0.2578131854534149, 0.3478696346282959], [0.23480947315692902, 0.0843145027756691, 0.08692729473114014, 0.3137372136116028]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.11977646499872208, 0.21320702135562897, 0.14060527086257935, 0.06935633718967438], [0.006631460040807724, 0.2403852343559265, 0.12097999453544617, 0.272177129983902], [0.42431706190109253, 0.021122237667441368, 0.2849246561527252, 0.21280157566070557], [0.42431706190109253, 0.021122237667441368, 0.2849246561527252, 0.21280157566070557], [0.09864906966686249, 0.4189547300338745, 0.3182961344718933, 0.3324736952781677]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82878af8922b46863353e8192e9a86b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a238abe22b53aa2d790a87ba5492869c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bc80eacd048af929fa911378bf25623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6e894825353c335b8bec84889131dcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63ec4cc02e28738a7e86184bcc7e3760(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a238abe22b53aa2d790a87ba5492869c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75c43c24d6de141762f4371844030d6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.431508332490921, 0.3163807988166809, 0.2813553214073181, 0.42060959339141846], [0.20145507156848907, 0.1395324319601059, 0.32148462533950806, 0.44821107387542725], [0.1956084966659546, 0.00535923708230257, 0.46844279766082764, 0.21772362291812897], [0.431508332490921, 0.3163807988166809, 0.2813553214073181, 0.42060959339141846], [0.48524898290634155, 0.006882299669086933, 0.3400641977787018, 0.2631310820579529], [0.4988814890384674, 0.0466485433280468, 0.2382136881351471, 0.10030604153871536], [0.48524898290634155, 0.006882299669086933, 0.3400641977787018, 0.2631310820579529]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.1268680989742279, 0.3940819203853607, 0.13702614605426788, 0.24026523530483246], [0.24064131081104279, 0.11957230418920517, 0.3499946892261505, 0.48193952441215515], [0.21294118463993073, 0.2877182066440582, 0.11202623695135117, 0.3961676359176636], [0.1268680989742279, 0.3940819203853607, 0.13702614605426788, 0.24026523530483246], [0.42945852875709534, 0.3748156428337097, 0.3677677512168884, 0.013625810854136944], [0.25984281301498413, 0.2521388530731201, 0.14442987740039825, 0.397305965423584], [0.42945852875709534, 0.3748156428337097, 0.3677677512168884, 0.013625810854136944]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf8043fc38e4ed578df2e49bdfa58027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6fcc8a6a0d3ce689e80ac74a22d8d94f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.33181232213974]], [[0.06974681466817856]], [[0.4563100039958954]], [[0.40709802508354187]], [[0.10137591511011124]], [[0.21714185178279877]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7579535841941833]], [[0.5940682888031006]], [[0.5430742502212524]], [[0.573845624923706]], [[0.7277628183364868]], [[0.6181967258453369]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_c9337b4224820d1e41ed6d0d47432dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.17835448682308197]], [[0.4390904903411865]], [[0.30827003717422485]], [[0.45996543765068054]], [[0.18206551671028137]], [[0.3943363130092621]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6730959415435791]], [[0.6264330148696899]], [[0.724555253982544]], [[0.5223374366760254]], [[0.7338444590568542]], [[0.8147628903388977]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 23, 23, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45ae318d25bd69daa5f8c29c497acfa2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_606fc1ed2f155e568bafb3cfe4f199b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45ae318d25bd69daa5f8c29c497acfa2
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98d1aff2f76515745f752592ad9e32e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 11, 11, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2d587fdee5fed7b64f61590c5e64599(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18ea7c1c6fca4fb008767377029b08e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2d587fdee5fed7b64f61590c5e64599
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b85e1a08b1b27a2a968d617f720ed071(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12096, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 12096, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f2815a55edd859e045227802c19dca57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b85e1a08b1b27a2a968d617f720ed071
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 24, 24, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 42, 42, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a660020cb86e0398bf6a2e87a002dc2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2092c99fe6977d956c1c7c305d64dd61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a660020cb86e0398bf6a2e87a002dc2
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_758bf3801ed88050004599925b40e60d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a3b14ed238cb303541f682f377b346d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_758bf3801ed88050004599925b40e60d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.11059033870697021, 0.07338421791791916]], [[0.12060432136058807, 0.29075783491134644]], [[0.2402641475200653, 0.3665030002593994]], [[0.07487303018569946, 0.09319182485342026]], [[0.00415261322632432, 0.4588662385940552]], [[0.37556031346321106, 0.12091569602489471]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.1949375867843628, 0.19650663435459137]], [[0.029328227043151855, 0.49953198432922363]], [[0.19212405383586884, 0.4960818290710449]], [[0.413517564535141, 0.10393795371055603]], [[0.4010055661201477, 0.46761366724967957]], [[0.35190922021865845, 0.22220514714717865]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_d0497efe75b6222d29de68826aee0248(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_758bf3801ed88050004599925b40e60d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.043390195816755295, 0.022145669907331467]], [[0.1635356843471527, 0.41414403915405273]], [[0.09432101249694824, 0.048812177032232285]], [[0.11243130266666412, 0.03282276540994644]], [[0.34049803018569946, 0.03584573045372963]], [[0.23450057208538055, 0.37127581238746643]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.1949375867843628, 0.19650663435459137]], [[0.029328227043151855, 0.49953198432922363]], [[0.19212405383586884, 0.4960818290710449]], [[0.413517564535141, 0.10393795371055603]], [[0.4010055661201477, 0.46761366724967957]], [[0.35190922021865845, 0.22220514714717865]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    
    class PrimitiveOp_e04add1b8d6435475bb578528599f0c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 21824, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c044accb33e4f95633cd05986b983b80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e04add1b8d6435475bb578528599f0c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.48046234250068665, 0.22905594110488892]], [[0.2770739495754242, 0.3431546986103058]], [[0.24031347036361694, 0.48387277126312256]], [[0.4746449291706085, 0.4006763994693756]], [[0.10624674707651138, 0.05708790943026543]], [[0.49964481592178345, 0.2771390378475189]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    
    class PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 46, 46, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0ec4a74d6c67bc08e22f55b7a09eba1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='float32'),
                paddle.static.InputSpec(shape=[16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f9f48838693cb35d90d9de0d20cb937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ec4a74d6c67bc08e22f55b7a09eba1e
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.2276126593351364, 0.34858253598213196, 0.47927117347717285, 0.4129710793495178, 0.017648370936512947, 0.38352295756340027, 0.190532848238945, 0.46824896335601807, 0.18940813839435577, 0.2642250657081604, 0.2662023603916168, 0.47233644127845764, 0.2409619688987732, 0.2888906002044678, 0.39573538303375244, 0.3080037832260132], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_309c0a0ec7c8338855df5c19f032842d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ec4a74d6c67bc08e22f55b7a09eba1e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2276126593351364, 0.34858253598213196, 0.47927117347717285, 0.4129710793495178, 0.017648370936512947, 0.38352295756340027, 0.190532848238945, 0.46824896335601807, 0.18940813839435577, 0.2642250657081604, 0.2662023603916168, 0.47233644127845764, 0.2409619688987732, 0.2888906002044678, 0.39573538303375244, 0.3080037832260132], dtype='float32').reshape([16]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
            ]


    
    class PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 12, 12, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76c5cb57b95ec8abfdf740ca8bd2bfcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d8785fab36e42b156399b6fd6430c17
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76c5cb57b95ec8abfdf740ca8bd2bfcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d8785fab36e42b156399b6fd6430c17
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7465b9334dd44357509fde4e6b0e7821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2c070853b2eb7e652ce2f6a6d561ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 84, 84, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_662951437554d00be1190f47fb0d1a08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_efb6d835e93f9cbfc33c81da15cb2469(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_662951437554d00be1190f47fb0d1a08
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 38, 38, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b37699aed0f721da2fb41d1286ff1f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1696, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a4a216d2311b602152f67ac22d30a53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b37699aed0f721da2fb41d1286ff1f2
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1696, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daee33d3bec831e9aeea974f8b949df3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3a922f5cba6491a3337cc033f3b7d1
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7853e2e706e379f30edda700958b1b19(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3549, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3549, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76ceb5e3abfd26553394f0828637454d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7853e2e706e379f30edda700958b1b19
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1c007ce102cac4d9666a7625d0ee8862(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[3549, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8821e632a224f567a7a2db5037481771(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c007ce102cac4d9666a7625d0ee8862
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a4a216d2311b602152f67ac22d30a53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b37699aed0f721da2fb41d1286ff1f2
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ffd015ead0c7d8bdc8812ccd035a19b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25037649273872375, 0.04901019111275673, 0.029957201331853867, 0.3703627288341522], [0.4998129904270172, 0.20214392244815826, 0.4599210023880005, 0.23039565980434418], [0.4898363947868347, 0.01842341385781765, 0.3382553458213806, 0.48959681391716003], [0.2272232323884964, 0.34871217608451843, 0.3543879985809326, 0.38190561532974243], [0.29566892981529236, 0.25489845871925354, 0.12884438037872314, 0.22511659562587738]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.23842979967594147, 0.2781031131744385, 0.1888388842344284, 0.38554295897483826], [0.26523661613464355, 0.34089452028274536, 0.22515225410461426, 0.19126184284687042], [0.1216023787856102, 0.2251485288143158, 0.3785667419433594, 0.4806677997112274], [0.4923000931739807, 0.4137409031391144, 0.027203183621168137, 0.44860342144966125], [0.38564565777778625, 0.14669828116893768, 0.18097509443759918, 0.29368749260902405]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 48, 48, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 21, 21, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d8589af676c69297828101d52c58872(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5376, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5376, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca58f62326429a37d99fa33f1c1b7f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d8589af676c69297828101d52c58872
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4f734004b0f88ff2246a055126f3cd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20757071673870087, 0.005273854359984398, 0.16517065465450287, 0.18617504835128784], [0.3260256052017212, 0.20955143868923187, 0.40184587240219116, 0.18627451360225677], [0.0242769755423069, 0.11374565213918686, 0.2671873867511749, 0.18605898320674896], [0.3260256052017212, 0.20955143868923187, 0.40184587240219116, 0.18627451360225677], [0.0242769755423069, 0.11374565213918686, 0.2671873867511749, 0.18605898320674896]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.4676962196826935, 0.28657740354537964, 0.4662608802318573, 0.30012109875679016], [0.1984112560749054, 0.3507075607776642, 0.11108946800231934, 0.1902817189693451], [0.4010750949382782, 0.1335587352514267, 0.35621824860572815, 0.25092580914497375], [0.1984112560749054, 0.3507075607776642, 0.11108946800231934, 0.1902817189693451], [0.4010750949382782, 0.1335587352514267, 0.35621824860572815, 0.25092580914497375]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 44, 44, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5919a702a05ecbed77148e9a35e98c64(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 92, 92, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_47d629b24dffb314652b35d32950ae5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_108ed1d7c0481951e6441aa823f741a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.32824042439460754], [0.053689487278461456], [0.04877207800745964], [0.24639740586280823], [0.06132432818412781], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.33700039982795715], [0.1544954627752304], [0.3554002642631531], [0.4550243616104126], [0.38996535539627075], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_908ca087cec3ab7c746d9199d3a87ccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.23870912194252014], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.3334523141384125], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_a6ea45ebded6eb7507f11c930e3d5564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.32824042439460754], [0.11407893896102905], [0.04877207800745964], [0.24639740586280823], [0.4660728871822357], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.33700039982795715], [0.13599033653736115], [0.16089512407779694], [0.4550243616104126], [0.2522077262401581], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_814d22d1eec30130b3c450cbe73faf6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.477491170167923], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.2988268733024597], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_ea750094a997d34c63bdb06594093f7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.3819327652454376], [0.053689487278461456], [0.17333781719207764], [0.3545892834663391], [0.06132432818412781], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.1544954627752304], [0.3554002642631531], [0.06358063966035843], [0.38996535539627075], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_18d662cd7d72b155dd4f5a0c04556875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.23870912194252014], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.3334523141384125], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_0aa48723aaa2a08c3cd5faace75f2a55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08513391762971878], [-0.0048834290355443954], [0.09082716703414917], [0.06779509782791138], [-0.02920367568731308], [0.020875904709100723], [0.1249118521809578], [-0.005090379621833563], [0.07913517206907272]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.02909252792596817], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9abc8c17656cd27ad4b8aa527e38467a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.3819327652454376], [0.11407893896102905], [0.17333781719207764], [0.3545892834663391], [0.4660728871822357], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.13599033653736115], [0.16089512407779694], [0.06358063966035843], [0.2522077262401581], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_0039706d0dbc81b3592352a8edfcb091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.477491170167923], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.2988268733024597], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_78580359aa7d755a4652c80a66b8e8ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.034133654087781906], [0.0038358664605766535], [0.06032535061240196], [0.04406944662332535], [-0.018659166991710663], [-0.029324453324079514], [0.09581932425498962], [0.04879090562462807], [0.0004284780006855726]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.08513391762971878], [-0.0048834290355443954], [0.09082716703414917], [0.06779509782791138], [-0.02920367568731308], [0.020875904709100723], [0.09581932425498962], [-0.005090379621833563], [0.07913517206907272]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_ba6016583b3eef0389f5712854e6cd41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0], [-0.0], [0.0], [0.30361858010292053], [-0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[3.4941341876983643], [2.2730965614318848], [-0.5056218504905701], [-0.5383695960044861], [-0.5651114583015442], [1.7118940353393555], [0.0], [1.1043304204940796], [-183.68899536132812]], dtype='float32').reshape([9, 1]),
            ]


    
    class PrimitiveOp_3d2e955745c6e634f31b3d914bc0c0bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dacdeda3677d8a4dba3faaf9ead5120c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d2e955745c6e634f31b3d914bc0c0bc
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_251a593c42691a2543c26775324dfbfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.0004822202608920634]], [[0.40181925892829895]], [[0.4493107199668884]], [[0.1114136129617691]], [[0.1425887644290924]], [[0.28795698285102844]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.585241973400116]], [[0.7111523747444153]], [[0.7787365317344666]], [[0.7846488356590271]], [[0.5292868614196777]], [[0.5829117894172668]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_f9a7de722f1eabba3cbdefc408ffdbff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.29122838377952576]], [[0.4045177400112152]], [[0.12232992798089981]], [[0.43076565861701965]], [[0.004029393196105957]], [[0.3265424966812134]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.5337342023849487]], [[0.759053647518158]], [[0.5657528042793274]], [[0.7224377989768982]], [[0.5274229049682617]], [[0.5056599378585815]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_f2c9daa3e7a3b8a7b2b8f68f30e067a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e197e4f0679b9d3874d7589206b2297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2c9daa3e7a3b8a7b2b8f68f30e067a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5638e19d28915f88c9dfa8a1c20a4cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[5517, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82c3ba6f535eb740a0e23536042a7c86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5638e19d28915f88c9dfa8a1c20a4cf
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5517, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538a68770a9353ef3b447ae21cbfb6ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d065809d7d7d99be4d6ba2a3308d41a3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c23fa2877c2c167dbf7b84020455fd2c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11109, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 11109, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f92e245cb6a04394d4aa27081fac1086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23fa2877c2c167dbf7b84020455fd2c
        def get_inputs(self):
            return [
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2dd3af35a4e33a1ccdbbff46d3441d77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[11109, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92d627da6461529f6d99007924649411(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2dd3af35a4e33a1ccdbbff46d3441d77
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82c3ba6f535eb740a0e23536042a7c86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5638e19d28915f88c9dfa8a1c20a4cf
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bd341c26c429be4deceb7b7802859f50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25d151c0ab7bd26312863e64ef53ba19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd341c26c429be4deceb7b7802859f50
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42444175481796265, 0.42435893416404724, 0.32613885402679443, 0.17775212228298187], [0.4992370307445526, 0.014581345021724701, 0.47447216510772705, 0.19020026922225952], [0.03811301290988922, 0.19911915063858032, 0.02669600583612919, 0.4305139482021332], [0.4992370307445526, 0.014581345021724701, 0.47447216510772705, 0.19020026922225952], [0.03811301290988922, 0.19911915063858032, 0.02669600583612919, 0.4305139482021332], [0.3735699951648712, 0.3213905096054077, 0.09645315259695053, 0.05379442125558853], [0.3735699951648712, 0.3213905096054077, 0.09645315259695053, 0.05379442125558853]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.09077489376068115, 0.03478417918086052, 0.29913225769996643, 0.35426220297813416], [0.2355150431394577, 0.04002285748720169, 0.34607771039009094, 0.32934725284576416], [0.08519759029150009, 0.19337064027786255, 0.40454861521720886, 0.26980942487716675], [0.2355150431394577, 0.04002285748720169, 0.34607771039009094, 0.32934725284576416], [0.08519759029150009, 0.19337064027786255, 0.40454861521720886, 0.26980942487716675], [0.3071763813495636, 0.03489753603935242, 0.35671326518058777, 0.4256373643875122], [0.3071763813495636, 0.03489753603935242, 0.35671326518058777, 0.4256373643875122]], dtype='float32').reshape([7, 4]),
            ]


    
    class PrimitiveOp_2df025b8dc15866669770bb73d3d12b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='float32'),
                paddle.static.InputSpec(shape=[36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f39fa74d1388ccc45b534a701adcf1f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2df025b8dc15866669770bb73d3d12b7
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f39fa74d1388ccc45b534a701adcf1f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2df025b8dc15866669770bb73d3d12b7
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2cd5d46f83982b15093cad4f7d42ecac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_654d6ef0a569b160f7a9734ace2c7706(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2cd5d46f83982b15093cad4f7d42ecac
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19df51b8e3f9bc2bdce7babbcce3e567(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f959a8aff3b35dcd17492161700ccf18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df51b8e3f9bc2bdce7babbcce3e567
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_51b473ce00db4877225484272820f9f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_785873392b1b65d324ac1055bdf7c998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.04365863651037216, 0.14801615476608276, 0.059532683342695236, 0.1973104625940323, 0.10881417244672775], dtype='float32').reshape([6]),
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2f50cc972580b3918b2814a3df0a1478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.21491502225399017, 0.4360129237174988, 0.2119390070438385, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_fdcbd5be4452c696b29bbf7adb2bdb92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45263877511024475, 0.18129011988639832, 0.45884010195732117, 0.49139586091041565, 0.3470446765422821, 0.2975061535835266], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.20852655172348022, 0.4761844575405121, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2383dd5aaa76a4f7248e58e113c74274(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.35412517189979553, 0.2729951739311218, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.06498825550079346, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_445ef2f0e762ec4d8c78b2750d4c3491(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.18129011988639832, 0.4002115726470947, 0.3979283273220062, 0.3470446765422821, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.4002115726470947, 0.4761844575405121, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_30b91327a2d79dd34b432750a5195ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.35412517189979553, 0.2729951739311218, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.43007832765579224, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_488b26ff223d139ed8340a5de82b13c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e4b346a9e213c752a02b7e137ddc45e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.4360129237174988, 0.43007832765579224, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_cb641505622cab42920e26d45aa6b975(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.07526008784770966, 0.02078755758702755, -0.02064768597483635, 0.003164077177643776, 0.009952809661626816, 0.030713750049471855], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0, 0.0, -0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_be8a28f82e647b40ec37636a4bfac50b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1936129629611969, 0.2076808512210846, 0.27411386370658875, 0.22873049974441528, 0.3200559914112091, 0.1432904452085495], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4512457847595215, 0.33517807722091675, 0.3336833119392395, 0.48379015922546387, 0.37571316957473755, 0.20266865193843842], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_61cffe64829ead679e18604827d07256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.17710404098033905, 0.3038899004459381, 0.4278310537338257, 0.32100868225097656, 0.2117002010345459, 0.24121758341789246], dtype='float32').reshape([6]),
                paddle.to_tensor([0.26647061109542847, 0.4544703960418701, 0.3953688144683838, 0.16899171471595764, 0.40512338280677795, 0.41586926579475403], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f7f3f08a1102cf2f3a9a4984e2c7774c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45263877511024475, 0.37170305848121643, 0.45884010195732117, 0.49139586091041565, 0.44280150532722473, 0.2975061535835266], dtype='float32').reshape([6]),
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.20852655172348022, 0.3979283273220062, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_85344ac3955a2bf4dd070b789fd9c10c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.4360129237174988, 0.43007832765579224, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.06498825550079346, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_aa57ce83c31139612a7a692e33ac1ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.01364241074770689, 1.354771375656128, -1.252467393875122, 0.07299936562776566, 0.31902867555618286, 0.864149808883667], dtype='float32').reshape([6]),
                paddle.to_tensor([0.7114414572715759, 1.0737632513046265, -1.5060021877288818, 0.9982069134712219, -0.8928132057189941, -0.3794630169868469], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_e3b406efd084931e8fc6622b64eb8d02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1794, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c73ec9d63948ab237fa81a0760fee842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3b406efd084931e8fc6622b64eb8d02
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9615a829a382f6c1848ea47a20b655e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1794, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e2583529b0ba944ddcb195979a563c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9615a829a382f6c1848ea47a20b655e6
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76ceb5e3abfd26553394f0828637454d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7853e2e706e379f30edda700958b1b19
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8821e632a224f567a7a2db5037481771(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c007ce102cac4d9666a7625d0ee8862
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c73ec9d63948ab237fa81a0760fee842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3b406efd084931e8fc6622b64eb8d02
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f1fc4ff7304811addd6aa1e8c524bace(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b75f0c99eac6ad57fef8a32821a3fc37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1fc4ff7304811addd6aa1e8c524bace
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c9932fc45296674f2f74b8b06a5dade(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='float32'),
                paddle.static.InputSpec(shape=[24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb0fb4311c0d7d6cfdd2d7e2f7a6168a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c9932fc45296674f2f74b8b06a5dade
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.2594594955444336, 0.22706426680088043, 0.28697821497917175, 0.16908468306064606, 0.16174711287021637, 0.1938442438840866, 0.42815297842025757, 0.11389681696891785, 0.02597997337579727, 0.2570507228374481, 0.23708491027355194, 0.05830814689397812, 0.4899848997592926, 0.2215232402086258, 0.23221857845783234, 0.024046774953603745, 0.4813443124294281, 0.1871427595615387, 0.23423327505588531, 0.1138496994972229, 0.25865429639816284, 0.3344466984272003, 0.28236815333366394, 0.04511779919266701], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_5225752410fd50059b5891ae46b43723(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c9932fc45296674f2f74b8b06a5dade
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2594594955444336, 0.22706426680088043, 0.28697821497917175, 0.16908468306064606, 0.16174711287021637, 0.1938442438840866, 0.42815297842025757, 0.11389681696891785, 0.02597997337579727, 0.2570507228374481, 0.23708491027355194, 0.05830814689397812, 0.4899848997592926, 0.2215232402086258, 0.23221857845783234, 0.024046774953603745, 0.4813443124294281, 0.1871427595615387, 0.23423327505588531, 0.1138496994972229, 0.25865429639816284, 0.3344466984272003, 0.28236815333366394, 0.04511779919266701], dtype='float32').reshape([24]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c794c1ac10698f9695867ead7fb0e1a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87adf1d030e2bcd2ba4245fdd2ceabec
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe23627d264f846ae3ab70ac54c61363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e48ddbb1c1e4a3a026472015a10731d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f0df63aff23adca843a83ead17e93aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1504, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4fcc55da29fe55993f7bbbefb3edd8b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f0df63aff23adca843a83ead17e93aa
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1504, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2a965b52ab4c13b49d9832543cd8e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c97e0b0d16abed4c7a96cb3f70bd7884
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fcf0d3df6b1ec1589ec5a555df0bb37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3024, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3024, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_111d6dc25e222b7eae0dbec53b66705d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fcf0d3df6b1ec1589ec5a555df0bb37
        def get_inputs(self):
            return [
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0c8c06c3eccdce0f7d6e05912386cb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[3024, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7845dc5718ae3ea58a7ffbee73293fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0c8c06c3eccdce0f7d6e05912386cb2
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fcc55da29fe55993f7bbbefb3edd8b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f0df63aff23adca843a83ead17e93aa
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24869505cb1a3492ce5a3f28112ee9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407dc0e905d944ce60bc3a246cf3819b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bb4a5f8d2f9250767a759fceb26fb2b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='float32'),
                paddle.static.InputSpec(shape=[4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7abc40ed183e0ec0c6b26f4736f940d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb4a5f8d2f9250767a759fceb26fb2b8
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.08582065254449844, 0.4129891097545624, 0.49300137162208557, 0.21711167693138123], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_1e79629f4b4cba9613f181f811b7e652(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb4a5f8d2f9250767a759fceb26fb2b8
        def get_inputs(self):
            return [
                paddle.to_tensor([0.08582065254449844, 0.4129891097545624, 0.49300137162208557, 0.21711167693138123], dtype='float32').reshape([4]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_633ca9186c88ae9ce23a470677ade55e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f090e4ebe749ea2cde6f0c3d397c7d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a9cf8bc37b9ebba6a0bdb54231ae6abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f090e4ebe749ea2cde6f0c3d397c7d0b
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_712d37a04c1b3335ed205b4c38775b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d1aff2f76515745f752592ad9e32e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_99e17b6d4406f6436b786e9839fcef53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ebe98945117f843fb0986f8a875fb76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e17b6d4406f6436b786e9839fcef53
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21024103462696075, 0.146906316280365, 0.3382836878299713, 0.3186817169189453], [0.30262491106987, 0.07154060900211334, 0.4640345871448517, 0.44525519013404846], [0.3060167729854584, 0.30472156405448914, 0.3321487009525299, 0.4837305247783661], [0.02388036996126175, 0.12797322869300842, 0.10076813399791718, 0.4382765293121338], [0.02388036996126175, 0.12797322869300842, 0.10076813399791718, 0.4382765293121338], [0.3060167729854584, 0.30472156405448914, 0.3321487009525299, 0.4837305247783661]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.36117079854011536, 0.3021794855594635, 0.3944045305252075, 0.16512134671211243], [0.37474554777145386, 0.4685516059398651, 0.3396102488040924, 0.0575605146586895], [0.10093643516302109, 0.15885069966316223, 0.27939414978027344, 0.3892151415348053], [0.15035083889961243, 0.31797781586647034, 0.42872291803359985, 0.4998945891857147], [0.15035083889961243, 0.31797781586647034, 0.42872291803359985, 0.4998945891857147], [0.10093643516302109, 0.15885069966316223, 0.27939414978027344, 0.3892151415348053]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_f2114a3737d3f797cc5b355761adf60d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4699752926826477, 0.20170211791992188, 0.33375540375709534, 0.3653546869754791], [0.09326086193323135, 0.3276481330394745, 0.1447921097278595, 0.30204591155052185], [0.2558746635913849, 0.3609501123428345, 0.29354995489120483, 0.09777697175741196], [0.23382103443145752, 0.29737725853919983, 0.19029706716537476, 0.21787863969802856], [0.4699752926826477, 0.20170211791992188, 0.33375540375709534, 0.3653546869754791]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.15726704895496368, 0.25012800097465515, 0.49190860986709595, 0.3168145418167114], [0.4363638758659363, 0.26189401745796204, 0.2109360694885254, 0.028239434584975243], [0.3192857503890991, 0.24466922879219055, 0.1895640790462494, 0.1258022040128708], [0.4110543727874756, 0.040166862308979034, 0.39998859167099, 0.031311191618442535], [0.15726704895496368, 0.25012800097465515, 0.49190860986709595, 0.3168145418167114]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_751c6067333dd291182750c7fcea6a74(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c76d974d52786c2465216f6eb0e27e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_751c6067333dd291182750c7fcea6a74
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c55035f7a062fc218d885b1c9b738341(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_144c5285921ff7c3525e9d08c1645742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8e4b75e7faa614202abdcaba91bb0d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_144c5285921ff7c3525e9d08c1645742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_3dcf32840d69fa62603402589daac805(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_0427003c01eb0d895686d6df8fb8acee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_7eba0b09fced869f83a8e7a7c7a50e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c0647f41d16a230522612cbe68410423(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0809708684682846]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_0427003c01eb0d895686d6df8fb8acee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_98218e5336b76f797ba1819b99262427(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_38f73b3209ddcefed505843ff18cc4b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08737017959356308]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0809708684682846]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_5921daa1f2a6de09030c564409cf1da7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0732436552643776]], dtype='float32').reshape([1, 1]),
            ]


    
    class PrimitiveOp_1783249d8b2fd3a8e22045537013db81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_199a879eff167da28f8e423cb7931eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.2049286961555481], [0.05052289366722107], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.37558823823928833], [0.3696064352989197], [0.3602122664451599], [0.49161165952682495], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_b102f2e8c885582eb8362714563835a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.4184499979019165], [0.4230078458786011], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ef10d26f032b72acc895cdb224a39c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.2049286961555481], [0.05052289366722107], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06345830112695694], [0.37558823823928833], [0.1739027202129364], [0.027879230678081512], [0.49161165952682495], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_12e9ed4d8f39abf7a27b276c9599f5ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.13699816167354584], [0.30248284339904785], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_bf9fefade8dbbf52268a9c55646cb560(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.38781505823135376], [0.12123280763626099], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.19266913831233978], [0.3696064352989197], [0.3602122664451599], [0.343851238489151], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d57b2c178f81c5b02fb1f503194a55c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.4184499979019165], [0.4230078458786011], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_6a43180ad87b098f56e306fbfbcb2f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.009171975776553154], [-0.00045348587445914745], [-0.13099893927574158], [-0.018913721665740013], [-0.02232457511126995], [0.0065928734838962555]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_9f2e2501e17ea6e7ffabdccb5eae754a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.38781505823135376], [0.12123280763626099], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06345830112695694], [0.19266913831233978], [0.1739027202129364], [0.027879230678081512], [0.343851238489151], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_13969e53207dbf0b843ba6241b5a1a57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.13699816167354584], [0.30248284339904785], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_6fb16947c5c427448a8fdda62825911a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.017743036150932312], [0.033636245876550674], [0.046311333775520325], [-0.0053473603911697865], [-0.02140507660806179], [-0.004702110309153795]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.009171975776553154], [-0.0004534857871476561], [-0.13099893927574158], [-0.018913721665740013], [-0.02232457511126995], [0.0065928734838962555]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_c2fefef3c001b1e7c5a1194889cd7715(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[1.516933798789978], [1.0134820938110352], [3.82865834236145], [-2.537020206451416], [-0.04295703023672104], [2.402109384536743]], dtype='float32').reshape([6, 1]),
            ]


    
    class PrimitiveOp_d7d91148da6f4d25d1ed84b8a21b1473(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_85151c91e58216005f5859deb948147a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7d91148da6f4d25d1ed84b8a21b1473
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29697567224502563, 0.033938176929950714, 0.12153984606266022, 0.16929808259010315], [0.4059440791606903, 0.4174965023994446, 0.09141463786363602, 0.08322188258171082], [0.15150173008441925, 0.09857954829931259, 0.04490986093878746, 0.2046736776828766], [0.31510642170906067, 0.3103889226913452, 0.10195562988519669, 0.23645362257957458]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.43735429644584656, 0.3411722481250763, 0.35462242364883423, 0.3983362317085266], [0.26059871912002563, 0.4778519570827484, 0.48979929089546204, 0.27005699276924133], [0.19344715774059296, 0.34565991163253784, 0.26136934757232666, 0.3304290473461151], [0.31872597336769104, 0.1560906022787094, 0.2340485155582428, 0.13768287003040314]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a03602a23722db5e3469f444f80b35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91170f7c7ac5b6d082788ac31dc0b26b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3b5a1fde69b8b80beab719e92ed863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b796f90c1efa2b281fbf9e37fa0c2b1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7eae067d2a668855b27c601d44cd4d8e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f9266b7d6a681f99d8072dae4a56c43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7eae067d2a668855b27c601d44cd4d8e
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48b70b2c59ddbe8700aa6426e7429af3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2039, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3319750c583582a6d7f8fb3ea84f15b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48b70b2c59ddbe8700aa6426e7429af3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2039, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f59173ac3f0739bb440ff87bf4f7f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18a16cc588eb0c1aa4a1c3ddf2742d63
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_63440ba422341bd8c041ac36ae284715(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4116, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4116, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81f489c8ef524240dd71d44abf6cf2a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63440ba422341bd8c041ac36ae284715
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe62324f52f899d16121b9dbd26035da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[4116, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0776572ec26624e4543cd1578b753dbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe62324f52f899d16121b9dbd26035da
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3319750c583582a6d7f8fb3ea84f15b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48b70b2c59ddbe8700aa6426e7429af3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64c7048f6de97fd300bb5b2052bc10d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd341c26c429be4deceb7b7802859f50
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18468241393566132, 0.33149173855781555, 0.24620825052261353, 0.16647128760814667], [0.18468241393566132, 0.33149173855781555, 0.24620825052261353, 0.16647128760814667], [0.4424126446247101, 0.48034167289733887, 0.41602855920791626, 0.4828370213508606], [0.37814199924468994, 0.462110310792923, 0.3353201150894165, 0.21955512464046478], [0.19167256355285645, 0.028036119416356087, 0.43969249725341797, 0.18763285875320435], [0.1682383418083191, 0.4110064208507538, 0.4892330467700958, 0.2316710650920868], [0.26976892352104187, 0.2744980752468109, 0.3080321252346039, 0.047989457845687866]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.35786253213882446, 0.32739314436912537, 0.3534325063228607, 0.4124147593975067], [0.35786253213882446, 0.32739314436912537, 0.3534325063228607, 0.4124147593975067], [0.16995151340961456, 0.3392655551433563, 0.06221455708146095, 0.45378729701042175], [0.33701011538505554, 0.21520060300827026, 0.3914770781993866, 0.127670019865036], [0.2887539267539978, 0.44512802362442017, 0.24220089614391327, 0.03608894720673561], [0.19074024260044098, 0.1567150354385376, 0.47922074794769287, 0.05587359517812729], [0.2514936625957489, 0.15642105042934418, 0.3574431538581848, 0.026684166863560677]], dtype='float32').reshape([7, 4]),
            ]


    
    class PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 22, 22, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a7f59500bad0a3932f95735168c5b57d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3cba1fe3e61bca9928aaba2e1ba1d96a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7f59500bad0a3932f95735168c5b57d
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0e6ce9f4fb148108ff0e761602768e60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a27ef90ed8cb64721c2b2fd1d3d7294(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e6ce9f4fb148108ff0e761602768e60
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a860bd928bebda9db0cd1067211fd8ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[4584, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06f72d378a64d8fbd3959cc6de492705(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a860bd928bebda9db0cd1067211fd8ea
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4584, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7c08718e0ee2cef0e99de8e626be69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23e96fa3eec2556d2eb58839f5e7302
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e0efe1c96201f5af52d61fa5c3904f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[9261, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 9261, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb2b9b6f8d1dfeb4d5bcf790b79a87f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e0efe1c96201f5af52d61fa5c3904f2
        def get_inputs(self):
            return [
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_254b5c21a121fc11637023c21af81060(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[9261, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a27b874ad8a339b9285747285658b81b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_254b5c21a121fc11637023c21af81060
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06f72d378a64d8fbd3959cc6de492705(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a860bd928bebda9db0cd1067211fd8ea
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a7b724710a8fb48d00e4ade1c31695eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1071, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_850ed771cdcf05a5d82ca8558cf01220(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7b724710a8fb48d00e4ade1c31695eb
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_28412cc55c55e30369049754030dc293(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1071, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f59eea0036f7f3281eb591352cb82ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28412cc55c55e30369049754030dc293
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc2fbb321bb21023c9c2bb9125a21b42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2100, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2100, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd35bd638b6e52c8a3e9f3e06ace018d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc2fbb321bb21023c9c2bb9125a21b42
        def get_inputs(self):
            return [
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ed00d9ea0f28dbe17ba33de0e8ba55ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[2100, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_020e3da8effd6c7b9e1ea31bc2b8da38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed00d9ea0f28dbe17ba33de0e8ba55ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850ed771cdcf05a5d82ca8558cf01220(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7b724710a8fb48d00e4ade1c31695eb
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be26e782edaf234d4ca1c7a28178877d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
                paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2184f1498e732b73b11c463f4a1eb89e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be26e782edaf234d4ca1c7a28178877d
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_793f16ef046e2bc874c4bc422fa19833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e17b6d4406f6436b786e9839fcef53
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09556698054075241, 0.062248557806015015, 0.48012056946754456, 0.1216694638133049], [0.4499690532684326, 0.3386240601539612, 0.13975663483142853, 0.35235655307769775], [0.4499690532684326, 0.3386240601539612, 0.13975663483142853, 0.35235655307769775], [0.3378070592880249, 0.048087358474731445, 0.4574849605560303, 0.04036188870668411], [0.17514236271381378, 0.24521946907043457, 0.13926689326763153, 0.0350252240896225], [0.1317417472600937, 0.24890102446079254, 0.07940025627613068, 0.47396957874298096]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.1827584058046341, 0.46928292512893677, 0.3330903947353363, 0.32664167881011963], [0.23554666340351105, 0.14961957931518555, 0.36159104108810425, 0.34157228469848633], [0.23554666340351105, 0.14961957931518555, 0.36159104108810425, 0.34157228469848633], [0.133694127202034, 0.19349321722984314, 0.06283392757177353, 0.011135056614875793], [0.2018437683582306, 0.08454585075378418, 0.3254218101501465, 0.44437113404273987], [0.46540915966033936, 0.167648583650589, 0.3530118465423584, 0.1341691017150879]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41a77d32d8d5ed397d855d30cf751505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_693b628d1e62b8d30cea1ba8bf87d1ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_50771e3bd26f22a3304e6295efdddf3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[100, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ff8138c4db6cc6ca3f0c5675d68f6a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50771e3bd26f22a3304e6295efdddf3e
        def get_inputs(self):
            return [
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c5195df1055a103146ed731f087a9e76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ec6867a3cfefcf7be770fd498b5f7c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5195df1055a103146ed731f087a9e76
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1.3719160556793213, 0.5977555513381958, 0.5020655393600464, 0.020979739725589752], [0.5314668416976929, 1.1458791494369507, 0.8261379599571228, 0.6206862330436707]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619e1e692c600e52cfe427174a38f603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92f64cf7ca0e2a044484fd91ed42eadf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2328bba53f4a14485acc4eaa9aa51f61(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd883c81ca0e3a6b67ede4bcf39f49a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2328bba53f4a14485acc4eaa9aa51f61
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2030603a607878eb7af3f4e178b675a8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d0d4c82617bf472b115ec91495577cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2030603a607878eb7af3f4e178b675a8
        def get_inputs(self):
            return [
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_de0324bed078bcbd34522cd4edfe7a49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_792b25f7b7d300c3cd89b5aea478825a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de0324bed078bcbd34522cd4edfe7a49
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3960612714290619, 0.04548724740743637, 0.1888345181941986, 10.237210273742676], [0.29948604106903076, 3.9570305347442627, 12.784720420837402, 6.013519287109375]], dtype='float32').reshape([2, 4]),
            ]


    
    class PrimitiveOp_792a19c7607a4198d6abfd687ef64154(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5cdce91f5ba1a22b1b9d3430cd4814c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.08265111595392227], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f2fb9f8938220233cdf471943c843051(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.03426346927881241], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b22387f58f682d2a2946cb73bc4c34d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.08265111595392227], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_50df64d6eaa13995b8b5ed4bad944a14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.3689388930797577], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_1222a2ebba9c506e0ce30bab76da0c0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.1922953873872757], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_4e5e3416bed3702095279f40014b9249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.03426346927881241], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b116a21fcebf9187e2e9dba7393eefc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015536606311798096], [-0.028670985251665115], [0.08992450684309006], [-0.0018401052802801132], [0.001199607620947063]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_245a64019634e156872e6a1f2d84b961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.1922953873872757], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_61bfe3b763725fcce3fb37f07c08a512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.3689388930797577], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_77a45103a41e9b70ff459d0631697e8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.024802792817354202], [0.09054671227931976], [0.1189025491476059], [-0.006836474873125553], [0.0045682224445044994]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.015536606311798096], [-0.028670985251665115], [0.08992450684309006], [-0.0018401051638647914], [0.0011996077373623848]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f950d2c5010d6a6857c7bdbccc1eabc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[1.6264054775238037], [1.3166429996490479], [0.24371254444122314], [0.7308400273323059], [0.7374016642570496]], dtype='float32').reshape([5, 1]),
            ]


    
    class PrimitiveOp_2bb153a0a2e16ed70614d8dbb0915c4f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_152997e4c92adc5308ff687824726aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bb153a0a2e16ed70614d8dbb0915c4f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70e9c6804d658ec7573924d5a4675506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8dcdebd9d06b848d2b43b5896c10191
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dda1787b53eef3f7890316bda352982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5919a702a05ecbed77148e9a35e98c64
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 19, 19, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7e632f845b8ffadbf8f0ed10b8fb9842(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2370, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5069589dbe3159eb96f2de65b77b5751(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e632f845b8ffadbf8f0ed10b8fb9842
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_43921ff7ae38ccfda436d0dc95158006(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2370, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2f38beedc701d274e5591757d9f17a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43921ff7ae38ccfda436d0dc95158006
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0c5ed65ce7853aa472d1aa95446a6df9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4725, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4725, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7142c611ab3a959970bbc54c77bd5c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c5ed65ce7853aa472d1aa95446a6df9
        def get_inputs(self):
            return [
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_41104e95dfbaf12aefa674d2a2dc731e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[4725, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42757bac24015174b9c86e768575c19e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41104e95dfbaf12aefa674d2a2dc731e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5069589dbe3159eb96f2de65b77b5751(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e632f845b8ffadbf8f0ed10b8fb9842
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a9f4535e64f19099386e177518595290(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2993, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5018fa18a5575056c3391fcb0e6091b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9f4535e64f19099386e177518595290
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74361551a82756a1e8cf126c3b343a67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2993, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7afe82aa48752449401ed4cad33559f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74361551a82756a1e8cf126c3b343a67
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba32373d516b77e215ce3096789dee24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_761ca360ff434b5cd6082110415334ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba32373d516b77e215ce3096789dee24
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ee9ad3d17e4e2e033fba4ef918d3e1f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88f0ed15dabd97e38372d022bc9259e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ee9ad3d17e4e2e033fba4ef918d3e1f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5018fa18a5575056c3391fcb0e6091b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9f4535e64f19099386e177518595290
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ac8fbd0a7a33e6712f8ea35ec9421f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[3832, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e27ae833b2dafbc1374b33599846ab2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ac8fbd0a7a33e6712f8ea35ec9421f1
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3832, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90367e79edbbdf6ec61e79311ce53164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067cd5ed0bb2b0f78867b718b8404b9
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3094fd9d23b90fe05d505c18ee3e1b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7581, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 7581, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c3022baa2199764559a2c9c9756c855(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3094fd9d23b90fe05d505c18ee3e1b4
        def get_inputs(self):
            return [
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9edbf6a470c67f5178e1c2d4e8dc44ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[7581, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ec426282b918f077541a659eee0b0fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9edbf6a470c67f5178e1c2d4e8dc44ff
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27ae833b2dafbc1374b33599846ab2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ac8fbd0a7a33e6712f8ea35ec9421f1
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_60fac92c414d9e79c103bcbd760cbbcf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6615cf489be19d50df018cc4e6fcd32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60fac92c414d9e79c103bcbd760cbbcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_01619829b918ef10f2f7b192c49cd435(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ddf2f5cecbd424e57f065d31703dcf71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01619829b918ef10f2f7b192c49cd435
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c56cd7b87b590bc893af8bc750466e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b001fe4fea7f5f4b97a893afa763dbd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95dda2536695e65bc3b2ac2ccc46e0a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_176569c56389456cdab59dd17f25917b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95dda2536695e65bc3b2ac2ccc46e0a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e6ee6c518cc2c17722e81ecf9fb65043(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='float32'),
                paddle.static.InputSpec(shape=[20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_85e4438e4cfb15933e2206a492fcc41f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ee6c518cc2c17722e81ecf9fb65043
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.30421602725982666, 0.054143913090229034, 0.394016832113266, 0.46182653307914734, 0.3547346591949463, 0.1406528353691101, 0.37224218249320984, 0.43492934107780457, 0.23447169363498688, 0.12492112815380096, 0.23746684193611145, 0.47617754340171814, 0.04644746333360672, 0.10742539912462234, 0.4697968661785126, 0.29846274852752686, 0.3385973274707794, 0.021797627210617065, 0.2388172298669815, 0.046391844749450684], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_ea9c3ac3468fa84bf7f62efffec65189(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ee6c518cc2c17722e81ecf9fb65043
        def get_inputs(self):
            return [
                paddle.to_tensor([0.30421602725982666, 0.054143913090229034, 0.394016832113266, 0.46182653307914734, 0.3547346591949463, 0.1406528353691101, 0.37224218249320984, 0.43492934107780457, 0.23447169363498688, 0.12492112815380096, 0.23746684193611145, 0.47617754340171814, 0.04644746333360672, 0.10742539912462234, 0.4697968661785126, 0.29846274852752686, 0.3385973274707794, 0.021797627210617065, 0.2388172298669815, 0.046391844749450684], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
            ]


    
    class PrimitiveOp_c5b6d3e6784913d92fda481865d163c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60359fb5504d9f96f26c7108423e29a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.2982136011123657], [0.18588005006313324], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41876405477523804], [0.462017297744751], [0.11372362822294235], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_8a11f981e9187eafc8b2ed8b17d0315e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33433759212493896], [0.025396505370736122], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_78f2bf85282ebf755f0fa1d36341502c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.43085747957229614], [0.18588005006313324], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41876405477523804], [0.0375184640288353], [0.09816955029964447], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9d29301812abfe25cc946e545bd8f69d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.025396505370736122], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_539b58ef59c8b09e7c2bd6933b6b81a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4131563901901245], [0.2982136011123657], [0.22558467090129852], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.462017297744751], [0.11372362822294235], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_2feadf89d25f39f3f7e4cf7aa075cfc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33433759212493896], [0.12095697224140167], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_820f53f0d85e0a13cd7b72657df42a8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06323745846748352], [-0.025388313457369804], [0.017503943294286728], [0.007693326100707054]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_2d5c8e868bc2c9445b67935adb1c25a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4131563901901245], [0.43085747957229614], [0.22558467090129852], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.0375184640288353], [0.09816955029964447], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_edcb70743e53ec2fb91827942849d617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.12095697224140167], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_1484b379bd634b9e8d1e33d02b8321d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.026258420199155807], [0.011381367221474648], [0.030490899458527565], [-0.0003965869836974889]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.06323745846748352], [-0.025388313457369804], [0.017503943294286728], [0.007693326100707054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9fdc211617600e1e17cb2914c6bd4460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[3.408273696899414], [3.2306909561157227], [0.42592892050743103], [20.398836135864258]], dtype='float32').reshape([4, 1]),
            ]


    
    class PrimitiveOp_52cf55e9e92ea8b1e5d0defc13ba0a4f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d2d3b985c8237f3aa96100b6f759322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52cf55e9e92ea8b1e5d0defc13ba0a4f
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_feb8086152557fe77643482c36e323b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c66a2b00162f54ecf74bacb90e73d465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb8086152557fe77643482c36e323b3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a33b7c65669437df8eaa0a0821653bac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1995, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89970bbb73ecebaa457e6f5b77a939d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a33b7c65669437df8eaa0a0821653bac
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81f489c8ef524240dd71d44abf6cf2a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63440ba422341bd8c041ac36ae284715
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0776572ec26624e4543cd1578b753dbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe62324f52f899d16121b9dbd26035da
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c66a2b00162f54ecf74bacb90e73d465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_feb8086152557fe77643482c36e323b3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_176569c56389456cdab59dd17f25917b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95dda2536695e65bc3b2ac2ccc46e0a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3d0f95610020a3ca8a0fefc41c9b731e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b213080d71b95d71c4eacf01a15ff68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d0f95610020a3ca8a0fefc41c9b731e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c2119757f0fb398ba25b01b31a8e1b51(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6804, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6804, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b299afe0650b2176663890cb9c353ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2119757f0fb398ba25b01b31a8e1b51
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6779b7f1ec56099655e7dd1ae854793e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.059797223657369614, 0.3857690691947937, 0.4381629526615143, 0.1027114987373352], [0.4444314241409302, 0.43357622623443604, 0.13710230588912964, 0.45452260971069336], [0.24993541836738586, 0.26625725626945496, 0.2578131854534149, 0.3478696346282959], [0.24993541836738586, 0.26625725626945496, 0.2578131854534149, 0.3478696346282959], [0.23480947315692902, 0.0843145027756691, 0.08692729473114014, 0.3137372136116028]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.11977646499872208, 0.21320702135562897, 0.14060527086257935, 0.06935633718967438], [0.006631460040807724, 0.2403852343559265, 0.12097999453544617, 0.272177129983902], [0.42431706190109253, 0.021122237667441368, 0.2849246561527252, 0.21280157566070557], [0.42431706190109253, 0.021122237667441368, 0.2849246561527252, 0.21280157566070557], [0.09864906966686249, 0.4189547300338745, 0.3182961344718933, 0.3324736952781677]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed966464ac5075065eb08d8c7c255e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2703f42c7eb6d49fb70b1fe832ef9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72657202caa6fe415064df8975556ee1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abe204d1ec7578d1ce71c887a74a61f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a909afa13edba78090f68bfbf3b36c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab8ab9ea25671ab7fd7d97a9e81f8ca6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a909afa13edba78090f68bfbf3b36c4
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 3, 76, 76, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_609e8a44a9d5b199413c7b679a21450b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[4181, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2738e456d0a068aa3de6a1fb8484e72e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_609e8a44a9d5b199413c7b679a21450b
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4181, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dd7a33bc8dc9aa4f338c087ff7ad432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cc901c0c0f0c8ff99aa1b28e318175c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b90206e75fa3c377aaa600fc9797be8e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1cee13d468518614765cf22f79dec74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b90206e75fa3c377aaa600fc9797be8e
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_489187a356ab6e5a16978564b883cf76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ba5a472c7dfa97734c3fdbd27e398dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_489187a356ab6e5a16978564b883cf76
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2738e456d0a068aa3de6a1fb8484e72e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_609e8a44a9d5b199413c7b679a21450b
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cf13779c530e6c7bb4eb4b587dbcbd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd341c26c429be4deceb7b7802859f50
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.431508332490921, 0.3163807988166809, 0.2813553214073181, 0.42060959339141846], [0.20145507156848907, 0.1395324319601059, 0.32148462533950806, 0.44821107387542725], [0.1956084966659546, 0.00535923708230257, 0.46844279766082764, 0.21772362291812897], [0.431508332490921, 0.3163807988166809, 0.2813553214073181, 0.42060959339141846], [0.48524898290634155, 0.006882299669086933, 0.3400641977787018, 0.2631310820579529], [0.4988814890384674, 0.0466485433280468, 0.2382136881351471, 0.10030604153871536], [0.48524898290634155, 0.006882299669086933, 0.3400641977787018, 0.2631310820579529]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.1268680989742279, 0.3940819203853607, 0.13702614605426788, 0.24026523530483246], [0.24064131081104279, 0.11957230418920517, 0.3499946892261505, 0.48193952441215515], [0.21294118463993073, 0.2877182066440582, 0.11202623695135117, 0.3961676359176636], [0.1268680989742279, 0.3940819203853607, 0.13702614605426788, 0.24026523530483246], [0.42945852875709534, 0.3748156428337097, 0.3677677512168884, 0.013625810854136944], [0.25984281301498413, 0.2521388530731201, 0.14442987740039825, 0.397305965423584], [0.42945852875709534, 0.3748156428337097, 0.3677677512168884, 0.013625810854136944]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c698219a04b34837b38fe8273ecfe593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a406b2b4abaf4b1a1faccb2cc6132d2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c1e790f45640381dc035b63e8d1e6024(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc7a801d4745555f950da3b701049e42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e790f45640381dc035b63e8d1e6024
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6b58699b49ff9231455c42b6ed66576(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.33181232213974]], [[0.06974681466817856]], [[0.4563100039958954]], [[0.40709802508354187]], [[0.10137591511011124]], [[0.21714185178279877]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7579535841941833]], [[0.5940682888031006]], [[0.5430742502212524]], [[0.573845624923706]], [[0.7277628183364868]], [[0.6181967258453369]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_e6c5ec80c69cd71e72a5fdc8c7c399e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.17835448682308197]], [[0.4390904903411865]], [[0.30827003717422485]], [[0.45996543765068054]], [[0.18206551671028137]], [[0.3943363130092621]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6730959415435791]], [[0.6264330148696899]], [[0.724555253982544]], [[0.5223374366760254]], [[0.7338444590568542]], [[0.8147628903388977]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2cf34139b4eef42412b3115d46f69e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5627d7089bc7ccf837f32569031cb2da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e403131c78b3cf1e4b4f8eb994a423a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_787f053952c37fbaeda5e3810907422b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22ed387fdf4b1a5791d0d375a6917ea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.11059033870697021, 0.07338421791791916]], [[0.12060432136058807, 0.29075783491134644]], [[0.2402641475200653, 0.3665030002593994]], [[0.07487303018569946, 0.09319182485342026]], [[0.00415261322632432, 0.4588662385940552]], [[0.37556031346321106, 0.12091569602489471]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.1949375867843628, 0.19650663435459137]], [[0.029328227043151855, 0.49953198432922363]], [[0.19212405383586884, 0.4960818290710449]], [[0.413517564535141, 0.10393795371055603]], [[0.4010055661201477, 0.46761366724967957]], [[0.35190922021865845, 0.22220514714717865]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_52f01ef85ccad259c6bdfd98b5735ebf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.043390195816755295, 0.022145669907331467]], [[0.1635356843471527, 0.41414403915405273]], [[0.09432101249694824, 0.048812177032232285]], [[0.11243130266666412, 0.03282276540994644]], [[0.34049803018569946, 0.03584573045372963]], [[0.23450057208538055, 0.37127581238746643]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.1949375867843628, 0.19650663435459137]], [[0.029328227043151855, 0.49953198432922363]], [[0.19212405383586884, 0.4960818290710449]], [[0.413517564535141, 0.10393795371055603]], [[0.4010055661201477, 0.46761366724967957]], [[0.35190922021865845, 0.22220514714717865]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_eb95fc1493e2635707680b7366af234d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.48046234250068665, 0.22905594110488892]], [[0.2770739495754242, 0.3431546986103058]], [[0.24031347036361694, 0.48387277126312256]], [[0.4746449291706085, 0.4006763994693756]], [[0.10624674707651138, 0.05708790943026543]], [[0.49964481592178345, 0.2771390378475189]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cd4ad177bf59e7b8182006c518bdcee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.2276126593351364, 0.34858253598213196, 0.47927117347717285, 0.4129710793495178, 0.017648370936512947, 0.38352295756340027, 0.190532848238945, 0.46824896335601807, 0.18940813839435577, 0.2642250657081604, 0.2662023603916168, 0.47233644127845764, 0.2409619688987732, 0.2888906002044678, 0.39573538303375244, 0.3080037832260132], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_56553316d0dbddf51670765857858e50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2276126593351364, 0.34858253598213196, 0.47927117347717285, 0.4129710793495178, 0.017648370936512947, 0.38352295756340027, 0.190532848238945, 0.46824896335601807, 0.18940813839435577, 0.2642250657081604, 0.2662023603916168, 0.47233644127845764, 0.2409619688987732, 0.2888906002044678, 0.39573538303375244, 0.3080037832260132], dtype='float32').reshape([16]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae564804bdc4d141d224e4b77df53827(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae564804bdc4d141d224e4b77df53827(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f0773a14c0f5946f0c03c7863d5eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63299c2e4d225192c2b0be218c743f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb06185c2d11cc2415cf3bd6802175df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf618cd9b92c547abd7f829642918d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee1c02db88ba5f2372a7802d533bba5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bc4636b3ea34ab0e9db2ad06ef24173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb06185c2d11cc2415cf3bd6802175df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0b7f661e3bf6a7a4daf8a03c5266065(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25037649273872375, 0.04901019111275673, 0.029957201331853867, 0.3703627288341522], [0.4998129904270172, 0.20214392244815826, 0.4599210023880005, 0.23039565980434418], [0.4898363947868347, 0.01842341385781765, 0.3382553458213806, 0.48959681391716003], [0.2272232323884964, 0.34871217608451843, 0.3543879985809326, 0.38190561532974243], [0.29566892981529236, 0.25489845871925354, 0.12884438037872314, 0.22511659562587738]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.23842979967594147, 0.2781031131744385, 0.1888388842344284, 0.38554295897483826], [0.26523661613464355, 0.34089452028274536, 0.22515225410461426, 0.19126184284687042], [0.1216023787856102, 0.2251485288143158, 0.3785667419433594, 0.4806677997112274], [0.4923000931739807, 0.4137409031391144, 0.027203183621168137, 0.44860342144966125], [0.38564565777778625, 0.14669828116893768, 0.18097509443759918, 0.29368749260902405]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3edc86d4f015ff1c81e9c0c10b6dd5fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b245b1bfb03b3cebd65892f188f4363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20757071673870087, 0.005273854359984398, 0.16517065465450287, 0.18617504835128784], [0.3260256052017212, 0.20955143868923187, 0.40184587240219116, 0.18627451360225677], [0.0242769755423069, 0.11374565213918686, 0.2671873867511749, 0.18605898320674896], [0.3260256052017212, 0.20955143868923187, 0.40184587240219116, 0.18627451360225677], [0.0242769755423069, 0.11374565213918686, 0.2671873867511749, 0.18605898320674896]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.4676962196826935, 0.28657740354537964, 0.4662608802318573, 0.30012109875679016], [0.1984112560749054, 0.3507075607776642, 0.11108946800231934, 0.1902817189693451], [0.4010750949382782, 0.1335587352514267, 0.35621824860572815, 0.25092580914497375], [0.1984112560749054, 0.3507075607776642, 0.11108946800231934, 0.1902817189693451], [0.4010750949382782, 0.1335587352514267, 0.35621824860572815, 0.25092580914497375]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2fa0793049a656443ce9fe67b9e7eff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.32824042439460754], [0.053689487278461456], [0.04877207800745964], [0.24639740586280823], [0.06132432818412781], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.33700039982795715], [0.1544954627752304], [0.3554002642631531], [0.4550243616104126], [0.38996535539627075], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_de7aa4191288abc9f79ec0b5d54134a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.23870912194252014], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.3334523141384125], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2dad2ddf13ef0849e8a5b569a71122a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.32824042439460754], [0.11407893896102905], [0.04877207800745964], [0.24639740586280823], [0.4660728871822357], [0.06258141249418259]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.33700039982795715], [0.13599033653736115], [0.16089512407779694], [0.4550243616104126], [0.2522077262401581], [0.41581976413726807], [0.06474526226520538], [0.22361089289188385], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_97de8052ca91eda5001ce1ba5ebf7798(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07969757914543152], [0.14082278311252594], [0.477491170167923], [0.15222904086112976], [0.2556220293045044], [0.2010723203420639], [0.2752452790737152], [0.257482647895813], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.17808885872364044], [0.1587141752243042], [0.2988268733024597], [0.33936357498168945], [0.37306493520736694], [0.3378402590751648], [0.11509011685848236], [0.29118287563323975], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_980565ffac6c1676132c3d4894001908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10231184214353561], [0.21500952541828156], [0.033458199352025986], [0.3819327652454376], [0.053689487278461456], [0.17333781719207764], [0.3545892834663391], [0.06132432818412781], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.1544954627752304], [0.3554002642631531], [0.06358063966035843], [0.38996535539627075], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.4961751103401184]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9c8877c82efefe2bdf6d2f6db35137e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.23870912194252014], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.2503066658973694]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.3334523141384125], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.44735172390937805]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_77f5a15eef1222a60aa9b608abcf1b8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08513391762971878], [-0.0048834290355443954], [0.09082716703414917], [0.06779509782791138], [-0.02920367568731308], [0.020875904709100723], [0.1249118521809578], [-0.005090379621833563], [0.07913517206907272]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.02909252792596817], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2906250128bacd6817ee26723e5dac7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40283963084220886], [0.45022642612457275], [0.4985415041446686], [0.3819327652454376], [0.11407893896102905], [0.17333781719207764], [0.3545892834663391], [0.4660728871822357], [0.08616641163825989]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31189051270484924], [0.13599033653736115], [0.16089512407779694], [0.06358063966035843], [0.2522077262401581], [0.37870699167251587], [0.02283250354230404], [0.03404628112912178], [0.08131606876850128]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2c03fe92aa0efedc9d68d3c3cd6ccf27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4645979106426239], [0.151620551943779], [0.477491170167923], [0.4139777421951294], [0.40845590829849243], [0.21291503310203552], [0.35330283641815186], [0.264354944229126], [0.3754969537258148]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.08929294347763062], [0.13941359519958496], [0.2988268733024597], [0.2755478620529175], [0.273370623588562], [0.07012606412172318], [0.0644788146018982], [0.1514199823141098], [0.28715723752975464]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e7a553faeca3b83810d40ced3a408810(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.034133654087781906], [0.0038358664605766535], [0.06032535061240196], [0.04406944662332535], [-0.018659166991710663], [-0.029324453324079514], [0.09581932425498962], [0.04879090562462807], [0.0004284780006855726]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.08513391762971878], [-0.0048834290355443954], [0.09082716703414917], [0.06779509782791138], [-0.02920367568731308], [0.020875904709100723], [0.09581932425498962], [-0.005090379621833563], [0.07913517206907272]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_a6efec15ee9daa5333b107372c82822c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0], [-0.0], [0.0], [0.30361858010292053], [-0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[3.4941341876983643], [2.2730965614318848], [-0.5056218504905701], [-0.5383695960044861], [-0.5651114583015442], [1.7118940353393555], [0.0], [1.1043304204940796], [-183.68899536132812]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_1af58d553c5e7b6ec645021f13cf8a4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe2b1e6bb09aba51197669f46cd42c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.0004822202608920634]], [[0.40181925892829895]], [[0.4493107199668884]], [[0.1114136129617691]], [[0.1425887644290924]], [[0.28795698285102844]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.585241973400116]], [[0.7111523747444153]], [[0.7787365317344666]], [[0.7846488356590271]], [[0.5292868614196777]], [[0.5829117894172668]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_9c0cbb1777c11c3b5772b34b251a7fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.29122838377952576]], [[0.4045177400112152]], [[0.12232992798089981]], [[0.43076565861701965]], [[0.004029393196105957]], [[0.3265424966812134]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.5337342023849487]], [[0.759053647518158]], [[0.5657528042793274]], [[0.7224377989768982]], [[0.5274229049682617]], [[0.5056599378585815]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_3938bc10f243ced1b56a73317bb0e0ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9414b448d2da61066be3a0e193311883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e9d691c801a49c7d15013e3fd20d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_058d7920382cf156edd90d10cbccd897(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fb7e4bbce32cef9350027ea3a56b438(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9414b448d2da61066be3a0e193311883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ddef26f2e6f7fd18ffec68c34f21fc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42444175481796265, 0.42435893416404724, 0.32613885402679443, 0.17775212228298187], [0.4992370307445526, 0.014581345021724701, 0.47447216510772705, 0.19020026922225952], [0.03811301290988922, 0.19911915063858032, 0.02669600583612919, 0.4305139482021332], [0.4992370307445526, 0.014581345021724701, 0.47447216510772705, 0.19020026922225952], [0.03811301290988922, 0.19911915063858032, 0.02669600583612919, 0.4305139482021332], [0.3735699951648712, 0.3213905096054077, 0.09645315259695053, 0.05379442125558853], [0.3735699951648712, 0.3213905096054077, 0.09645315259695053, 0.05379442125558853]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.09077489376068115, 0.03478417918086052, 0.29913225769996643, 0.35426220297813416], [0.2355150431394577, 0.04002285748720169, 0.34607771039009094, 0.32934725284576416], [0.08519759029150009, 0.19337064027786255, 0.40454861521720886, 0.26980942487716675], [0.2355150431394577, 0.04002285748720169, 0.34607771039009094, 0.32934725284576416], [0.08519759029150009, 0.19337064027786255, 0.40454861521720886, 0.26980942487716675], [0.3071763813495636, 0.03489753603935242, 0.35671326518058777, 0.4256373643875122], [0.3071763813495636, 0.03489753603935242, 0.35671326518058777, 0.4256373643875122]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_cd4c7f79661bf690cbcf40118f518a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd4c7f79661bf690cbcf40118f518a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_224fddb1ca6622431af877bc9b3df646(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7098e869ec6bbcc523f1f9ce070f283(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b51bdd5f5a7dc1d7ec0f4402494c9f8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.04365863651037216, 0.14801615476608276, 0.059532683342695236, 0.1973104625940323, 0.10881417244672775], dtype='float32').reshape([6]),
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_54c874c04ebd91bbc75dc45a2a38f90e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.21491502225399017, 0.4360129237174988, 0.2119390070438385, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_537bb4e866226c3d115e2b89246e1751(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45263877511024475, 0.18129011988639832, 0.45884010195732117, 0.49139586091041565, 0.3470446765422821, 0.2975061535835266], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.20852655172348022, 0.4761844575405121, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b47630c56f9692ccf5222b02c91f0729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.35412517189979553, 0.2729951739311218, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.06498825550079346, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_72dab4f028cf1de1fe4ba35081607118(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.18129011988639832, 0.4002115726470947, 0.3979283273220062, 0.3470446765422821, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.4002115726470947, 0.4761844575405121, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_7e8b29fac1c23cf97dc7fd0e7df2f595(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.35412517189979553, 0.2729951739311218, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.43007832765579224, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_821772acec016f3984fc5b0f81ed8899(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.4002115726470947, 0.3979283273220062, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_69e07130dc0b1fa7a2ac6cb4124561bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.4360129237174988, 0.43007832765579224, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.43007832765579224, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e47dec68c91fa93ccd8bac8f72dbbf90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.07526008784770966, 0.02078755758702755, -0.02064768597483635, 0.003164077177643776, 0.009952809661626816, 0.030713750049471855], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0, 0.0, -0.0, 0.0, 0.0, -0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_fda384f0a99189c95d15b8a09d1851ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1936129629611969, 0.2076808512210846, 0.27411386370658875, 0.22873049974441528, 0.3200559914112091, 0.1432904452085495], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4512457847595215, 0.33517807722091675, 0.3336833119392395, 0.48379015922546387, 0.37571316957473755, 0.20266865193843842], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_fcaaaa3c0546e3c542e9319b9ed44bfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.17710404098033905, 0.3038899004459381, 0.4278310537338257, 0.32100868225097656, 0.2117002010345459, 0.24121758341789246], dtype='float32').reshape([6]),
                paddle.to_tensor([0.26647061109542847, 0.4544703960418701, 0.3953688144683838, 0.16899171471595764, 0.40512338280677795, 0.41586926579475403], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_64f870eef914bed48dce1f0099518e98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45263877511024475, 0.37170305848121643, 0.45884010195732117, 0.49139586091041565, 0.44280150532722473, 0.2975061535835266], dtype='float32').reshape([6]),
                paddle.to_tensor([0.06674034148454666, 0.37170305848121643, 0.20852655172348022, 0.3979283273220062, 0.404381662607193, 0.10783115774393082], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_6b2f50eaee499f1cfbfe527db2b4abd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3685716390609741, 0.420699805021286, 0.4360129237174988, 0.43007832765579224, 0.31833115220069885, 0.4968334138393402], dtype='float32').reshape([6]),
                paddle.to_tensor([0.029926525428891182, 0.39286479353904724, 0.419649213552475, 0.06498825550079346, 0.1128494143486023, 0.15476541221141815], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_7f7f67b6e90b7286d3fb0ad84e7e1353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.01364241074770689, 1.354771375656128, -1.252467393875122, 0.07299936562776566, 0.31902867555618286, 0.864149808883667], dtype='float32').reshape([6]),
                paddle.to_tensor([0.7114414572715759, 1.0737632513046265, -1.5060021877288818, 0.9982069134712219, -0.8928132057189941, -0.3794630169868469], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_3e3d539cbca9cd6f974048623f02b7e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18143af211d8aac59965b21b9dd9763a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee1c02db88ba5f2372a7802d533bba5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bc4636b3ea34ab0e9db2ad06ef24173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e3d539cbca9cd6f974048623f02b7e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3738802e3b2affa6656be927be90b8b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84fe50643aaf98f94858ab0985e2df5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.2594594955444336, 0.22706426680088043, 0.28697821497917175, 0.16908468306064606, 0.16174711287021637, 0.1938442438840866, 0.42815297842025757, 0.11389681696891785, 0.02597997337579727, 0.2570507228374481, 0.23708491027355194, 0.05830814689397812, 0.4899848997592926, 0.2215232402086258, 0.23221857845783234, 0.024046774953603745, 0.4813443124294281, 0.1871427595615387, 0.23423327505588531, 0.1138496994972229, 0.25865429639816284, 0.3344466984272003, 0.28236815333366394, 0.04511779919266701], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_1a641cb51fb3b8dcefc5f61669a075cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2594594955444336, 0.22706426680088043, 0.28697821497917175, 0.16908468306064606, 0.16174711287021637, 0.1938442438840866, 0.42815297842025757, 0.11389681696891785, 0.02597997337579727, 0.2570507228374481, 0.23708491027355194, 0.05830814689397812, 0.4899848997592926, 0.2215232402086258, 0.23221857845783234, 0.024046774953603745, 0.4813443124294281, 0.1871427595615387, 0.23423327505588531, 0.1138496994972229, 0.25865429639816284, 0.3344466984272003, 0.28236815333366394, 0.04511779919266701], dtype='float32').reshape([24]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8260b598a2ad22a346fe944254ed9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15388aba5388266de7b17d5fd158ec0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11af8709647196ce2099c5ce8f1918ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94229a01f116eafe9211b714d5e378c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aceac5bc2bf0c9efd501145333912648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4a362d86b9a93da5d962b9491cc8382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11af8709647196ce2099c5ce8f1918ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59309d1d9fee398679501cc3f9e97bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d678f25cbdc7c0c3059d5677c915c981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.08582065254449844, 0.4129891097545624, 0.49300137162208557, 0.21711167693138123], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_bf2ff873c46be6c59732aafac24a96f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.08582065254449844, 0.4129891097545624, 0.49300137162208557, 0.21711167693138123], dtype='float32').reshape([4]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
            ]


    
    class PrimitiveOp_e597a829678cda6f4ed07af0418aa2f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78a17ab815ab2339f01857e70b237711(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e597a829678cda6f4ed07af0418aa2f4
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_206c34cb6bcbf253871a8fa0d0529578(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e597a829678cda6f4ed07af0418aa2f4
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_820d66aa25b1cca39c9c14129db4dd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40899add8920b8687e8df0ac80468691(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21024103462696075, 0.146906316280365, 0.3382836878299713, 0.3186817169189453], [0.30262491106987, 0.07154060900211334, 0.4640345871448517, 0.44525519013404846], [0.3060167729854584, 0.30472156405448914, 0.3321487009525299, 0.4837305247783661], [0.02388036996126175, 0.12797322869300842, 0.10076813399791718, 0.4382765293121338], [0.02388036996126175, 0.12797322869300842, 0.10076813399791718, 0.4382765293121338], [0.3060167729854584, 0.30472156405448914, 0.3321487009525299, 0.4837305247783661]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.36117079854011536, 0.3021794855594635, 0.3944045305252075, 0.16512134671211243], [0.37474554777145386, 0.4685516059398651, 0.3396102488040924, 0.0575605146586895], [0.10093643516302109, 0.15885069966316223, 0.27939414978027344, 0.3892151415348053], [0.15035083889961243, 0.31797781586647034, 0.42872291803359985, 0.4998945891857147], [0.15035083889961243, 0.31797781586647034, 0.42872291803359985, 0.4998945891857147], [0.10093643516302109, 0.15885069966316223, 0.27939414978027344, 0.3892151415348053]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_b59aaf6531a893d26acef318cd7a62fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4699752926826477, 0.20170211791992188, 0.33375540375709534, 0.3653546869754791], [0.09326086193323135, 0.3276481330394745, 0.1447921097278595, 0.30204591155052185], [0.2558746635913849, 0.3609501123428345, 0.29354995489120483, 0.09777697175741196], [0.23382103443145752, 0.29737725853919983, 0.19029706716537476, 0.21787863969802856], [0.4699752926826477, 0.20170211791992188, 0.33375540375709534, 0.3653546869754791]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.15726704895496368, 0.25012800097465515, 0.49190860986709595, 0.3168145418167114], [0.4363638758659363, 0.26189401745796204, 0.2109360694885254, 0.028239434584975243], [0.3192857503890991, 0.24466922879219055, 0.1895640790462494, 0.1258022040128708], [0.4110543727874756, 0.040166862308979034, 0.39998859167099, 0.031311191618442535], [0.15726704895496368, 0.25012800097465515, 0.49190860986709595, 0.3168145418167114]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_73ece6d168a08bd3fa887a7c95aa72af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0434f312aa909d37a195006a970b602(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ae128115352692b1cbb0e8b72045fad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c0434f312aa909d37a195006a970b602(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20727156102657318]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.24695035815238953]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_79f7db25226187c456fb995314d4eeb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.27610716223716736]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_7ef18f9b7952128ac492f0ff155b148b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_cc62686e1541e14d7ddcb163e456a83e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3456944525241852]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_f35281470df6c37a86839b15b0749a1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0809708684682846]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_7ef18f9b7952128ac492f0ff155b148b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44695910811424255]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.1818336695432663]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c36dbf5612da382e55aba14835cfc40d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.35763055086135864]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.028087755665183067]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2fb1fbcce67d5f4f396b7da1b5f4f606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08737017959356308]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0809708684682846]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_e56a2a1ecb05cba4e6844f0307db8c1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0732436552643776]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ff1cb0d4a1e61abdb92aab88c95d533e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.2049286961555481], [0.05052289366722107], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.37558823823928833], [0.3696064352989197], [0.3602122664451599], [0.49161165952682495], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_548cfe56a14f6230797aff47d6adc21c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.4184499979019165], [0.4230078458786011], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_003c74cc17546b9ddd7948d0449ec6c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.2049286961555481], [0.05052289366722107], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06345830112695694], [0.37558823823928833], [0.1739027202129364], [0.027879230678081512], [0.49161165952682495], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_7bf502907ff01f454533bd926d6c287e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.1917470246553421], [0.0763665959239006], [0.3653966784477234], [0.06940227746963501]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.42717310786247253], [0.2651657462120056], [0.46184709668159485], [0.13699816167354584], [0.30248284339904785], [0.26277780532836914]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ea1c3a6bfd4fa7e120287b86741eb963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06397107243537903], [0.32089540362358093], [0.02683880925178528], [0.38781505823135376], [0.12123280763626099], [0.1445770114660263]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17218370735645294], [0.19266913831233978], [0.3696064352989197], [0.3602122664451599], [0.343851238489151], [0.4164700210094452]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e7e93a1e7bdcda73fe8cd23fbacd925a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07160773128271103], [0.0910111665725708], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.4184499979019165], [0.4230078458786011], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ba73f40234619a2da2c9b3cca78bdc2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.009171975776553154], [-0.00045348587445914745], [-0.13099893927574158], [-0.018913721665740013], [-0.02232457511126995], [0.0065928734838962555]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_03ba760056d0dfe25390c0717d3c4237(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1136389821767807], [0.3263870179653168], [0.4104270935058594], [0.38781505823135376], [0.12123280763626099], [0.18513862788677216]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06345830112695694], [0.19266913831233978], [0.1739027202129364], [0.027879230678081512], [0.343851238489151], [0.11910757422447205]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_5fb5a9bc112bf56674e0711bf87651c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3679124414920807], [0.3236524164676666], [0.3088380694389343], [0.12214173376560211], [0.39863425493240356], [0.18909907341003418]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.01432943344116211], [0.07210607081651688], [0.11303866654634476], [0.13699816167354584], [0.30248284339904785], [0.2603096663951874]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_0f464a7a2859d60b2ed207511e7c1e9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.017743036150932312], [0.033636245876550674], [0.046311333775520325], [-0.0053473603911697865], [-0.02140507660806179], [-0.004702110309153795]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.009171975776553154], [-0.0004534857871476561], [-0.13099893927574158], [-0.018913721665740013], [-0.02232457511126995], [0.0065928734838962555]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_976519ccf761e9799e557d922ff3d2c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [-0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[1.516933798789978], [1.0134820938110352], [3.82865834236145], [-2.537020206451416], [-0.04295703023672104], [2.402109384536743]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_56a60b59e3f52029d0df6606412d59de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29697567224502563, 0.033938176929950714, 0.12153984606266022, 0.16929808259010315], [0.4059440791606903, 0.4174965023994446, 0.09141463786363602, 0.08322188258171082], [0.15150173008441925, 0.09857954829931259, 0.04490986093878746, 0.2046736776828766], [0.31510642170906067, 0.3103889226913452, 0.10195562988519669, 0.23645362257957458]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.43735429644584656, 0.3411722481250763, 0.35462242364883423, 0.3983362317085266], [0.26059871912002563, 0.4778519570827484, 0.48979929089546204, 0.27005699276924133], [0.19344715774059296, 0.34565991163253784, 0.26136934757232666, 0.3304290473461151], [0.31872597336769104, 0.1560906022787094, 0.2340485155582428, 0.13768287003040314]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c929f78321da7b1c73d5df532f72babc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e2c210a19c12b0d2d667ac35a87004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd4ad40b9e6cb06109979ddd755a24b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05c66e9e80b77f7f718c855835bdb761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c49acc18fb6a078cb2e55a7012fd57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bbf7c0afdbea438b74c5eae4bcac9b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f64cb6fd9abf54063d21adfc7894a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05c66e9e80b77f7f718c855835bdb761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157d5371e1c02a2020f5d0d1c79c9c7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18468241393566132, 0.33149173855781555, 0.24620825052261353, 0.16647128760814667], [0.18468241393566132, 0.33149173855781555, 0.24620825052261353, 0.16647128760814667], [0.4424126446247101, 0.48034167289733887, 0.41602855920791626, 0.4828370213508606], [0.37814199924468994, 0.462110310792923, 0.3353201150894165, 0.21955512464046478], [0.19167256355285645, 0.028036119416356087, 0.43969249725341797, 0.18763285875320435], [0.1682383418083191, 0.4110064208507538, 0.4892330467700958, 0.2316710650920868], [0.26976892352104187, 0.2744980752468109, 0.3080321252346039, 0.047989457845687866]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.35786253213882446, 0.32739314436912537, 0.3534325063228607, 0.4124147593975067], [0.35786253213882446, 0.32739314436912537, 0.3534325063228607, 0.4124147593975067], [0.16995151340961456, 0.3392655551433563, 0.06221455708146095, 0.45378729701042175], [0.33701011538505554, 0.21520060300827026, 0.3914770781993866, 0.127670019865036], [0.2887539267539978, 0.44512802362442017, 0.24220089614391327, 0.03608894720673561], [0.19074024260044098, 0.1567150354385376, 0.47922074794769287, 0.05587359517812729], [0.2514936625957489, 0.15642105042934418, 0.3574431538581848, 0.026684166863560677]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c089057d08b7de722c2e54697878df9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdddeda0a869bfe102bd833a9e99a7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75765796aa18b71eed8a8e8836027f58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe7728ba199e5827745caf5257c6f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8417cbf4ddbe166177b4e5bea59889fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48e9a5db03edbdd8a81a2d8d42dd6bc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75765796aa18b71eed8a8e8836027f58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47951774a6947b7ff9b24662d1496290(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1646e3a638a17576f38fb7a026515889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77fee8356a1b0de98ec213ed503fdeb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bc0c3dc8045c74802061533aeb59dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47951774a6947b7ff9b24662d1496290(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a89dfe01d48a19b39a14b931a6f22cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_573195c405a55b5a52c0ff671e9d5e17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09556698054075241, 0.062248557806015015, 0.48012056946754456, 0.1216694638133049], [0.4499690532684326, 0.3386240601539612, 0.13975663483142853, 0.35235655307769775], [0.4499690532684326, 0.3386240601539612, 0.13975663483142853, 0.35235655307769775], [0.3378070592880249, 0.048087358474731445, 0.4574849605560303, 0.04036188870668411], [0.17514236271381378, 0.24521946907043457, 0.13926689326763153, 0.0350252240896225], [0.1317417472600937, 0.24890102446079254, 0.07940025627613068, 0.47396957874298096]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.1827584058046341, 0.46928292512893677, 0.3330903947353363, 0.32664167881011963], [0.23554666340351105, 0.14961957931518555, 0.36159104108810425, 0.34157228469848633], [0.23554666340351105, 0.14961957931518555, 0.36159104108810425, 0.34157228469848633], [0.133694127202034, 0.19349321722984314, 0.06283392757177353, 0.011135056614875793], [0.2018437683582306, 0.08454585075378418, 0.3254218101501465, 0.44437113404273987], [0.46540915966033936, 0.167648583650589, 0.3530118465423584, 0.1341691017150879]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0eff97c7946ec3fbd138d595a810b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d54b4c0274255a4200a257626b103198(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f11913d30e0936216370c991bda3b964(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1.3719160556793213, 0.5977555513381958, 0.5020655393600464, 0.020979739725589752], [0.5314668416976929, 1.1458791494369507, 0.8261379599571228, 0.6206862330436707]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_243f8cca02e18eabcc8631e9c35ab721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28af863461890eda4f89ebc15f89a065(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_321485876b504aa24b6a10287956155c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbc718d902efaaa1b335fd6198de5e99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3960612714290619, 0.04548724740743637, 0.1888345181941986, 10.237210273742676], [0.29948604106903076, 3.9570305347442627, 12.784720420837402, 6.013519287109375]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_a3b7dfba983dcbb80ed2682e3741967a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.08265111595392227], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b8af055c6563e68c27a2cb3674ea1f1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.03426346927881241], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_057d9c8d63faa8ba2459d2aa3a8423e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.08265111595392227], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.1261482983827591]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b34af14cfc4f9e102e017a8eaf61d8f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05733131244778633], [0.1570308655500412], [0.3689388930797577], [0.17223645746707916], [0.2121700644493103]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.446043998003006], [0.12132035940885544], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2bf2e7effa2650c4498074603be12a14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.104369156062603], [0.09522414207458496], [0.19618113338947296], [0.1922953873872757], [0.1039789542555809]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3746805787086487], [0.2069297879934311], [0.08722388744354248], [0.3281881809234619], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_44d7271673afe3b63d738e1f67999f3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.03426346927881241], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.30022138357162476], [0.40387892723083496], [0.24005642533302307]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_869014e54dce68fc00b3d66f920dbda9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015536606311798096], [-0.028670985251665115], [0.08992450684309006], [-0.0018401052802801132], [0.001199607620947063]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_003ba39979bea7bf26a9ebb77ec67a30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1386314034461975], [0.3368445634841919], [0.4964533746242523], [0.1922953873872757], [0.13507212698459625]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.29145920276641846], [0.07165557891130447], [0.07205324620008469], [0.22017629444599152], [0.10755358636379242]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e5b3b6361b8952548e91debe24441ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4198094606399536], [0.43931514024734497], [0.3689388930797577], [0.41905465722084045], [0.24269208312034607]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.2575170397758484], [0.09787295013666153], [0.0887727364897728], [0.17385192215442657], [0.07668683677911758]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_91b25ca2604d1f0a2368cb8a6bc783bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.024802792817354202], [0.09054671227931976], [0.1189025491476059], [-0.006836474873125553], [0.0045682224445044994]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.015536606311798096], [-0.028670985251665115], [0.08992450684309006], [-0.0018401051638647914], [0.0011996077373623848]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c6669f96152f8a3c4fbd06ba0146d7e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[1.6264054775238037], [1.3166429996490479], [0.24371254444122314], [0.7308400273323059], [0.7374016642570496]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c4c82d58b3c1de4f77d6df41da4c1239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af5c54f2ff01e1517af8605167586ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e011b007b994cc9c7e58b759d67f99e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ef33354ee835a7196e9a1b0c20f8930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_195bc68a3b5e1a8a35a554518eed7b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a2cbc4a6e4fd6a507c47523b68f2aa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd02780250d0b27ca48abce8f252bf24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ef33354ee835a7196e9a1b0c20f8930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5b09d70141ad5efb3de12d872559b61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4757296bcd5f334de845b93b0ae19b3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca761e7333f233d8f0e4e90a9557a97f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4225640db7d7743b5c47ae2e609b9362(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5b09d70141ad5efb3de12d872559b61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d63d773cddc975ec5e8cc29e08386155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e324e2127ffb632afc0287f8eb5cc3ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b00f4c34395a6755ddb8c2956603745(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef54c2eefbb3f8526e1cdbfdb990ee43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d63d773cddc975ec5e8cc29e08386155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24209c3231c88f98cf77095f7134913d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b01d14df83ac09763ab30092653478d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_000a4846d5f9d7d8265888b183b48cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b64b2d0be6f55b93150ae0072dbd4bc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db1bdf21fe9070524e3637f533cdfb60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.30421602725982666, 0.054143913090229034, 0.394016832113266, 0.46182653307914734, 0.3547346591949463, 0.1406528353691101, 0.37224218249320984, 0.43492934107780457, 0.23447169363498688, 0.12492112815380096, 0.23746684193611145, 0.47617754340171814, 0.04644746333360672, 0.10742539912462234, 0.4697968661785126, 0.29846274852752686, 0.3385973274707794, 0.021797627210617065, 0.2388172298669815, 0.046391844749450684], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_3ab5dc5768e0c7e68c916c9ee4ab131c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.30421602725982666, 0.054143913090229034, 0.394016832113266, 0.46182653307914734, 0.3547346591949463, 0.1406528353691101, 0.37224218249320984, 0.43492934107780457, 0.23447169363498688, 0.12492112815380096, 0.23746684193611145, 0.47617754340171814, 0.04644746333360672, 0.10742539912462234, 0.4697968661785126, 0.29846274852752686, 0.3385973274707794, 0.021797627210617065, 0.2388172298669815, 0.046391844749450684], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_ae0f530b28ef9cba57911dd9973099f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.2982136011123657], [0.18588005006313324], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41876405477523804], [0.462017297744751], [0.11372362822294235], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_fd1e882062072551b01464359032bb0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33433759212493896], [0.025396505370736122], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_91a0f4863c2fd241ccc5991e70328361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10784570127725601], [0.43085747957229614], [0.18588005006313324], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41876405477523804], [0.0375184640288353], [0.09816955029964447], [0.42315077781677246]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_33b098216120e988fce1cfa49a08c40a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.025396505370736122], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2566831707954407]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7268e014e110161e11148b2812106119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4131563901901245], [0.2982136011123657], [0.22558467090129852], [0.03406929969787598]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.462017297744751], [0.11372362822294235], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_eaf912e4dd458073f7b3a70dc35a1974(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33433759212493896], [0.12095697224140167], [0.2845987379550934], [0.09898775070905685]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2604764997959137], [0.1259506195783615], [0.3157579004764557], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_ce9ee6b82614ae963ed24390e3517704(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06323745846748352], [-0.025388313457369804], [0.017503943294286728], [0.007693326100707054]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_de6faedb6066058e93ae514cf20ce6b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4131563901901245], [0.43085747957229614], [0.22558467090129852], [0.29377901554107666]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2992156445980072], [0.0375184640288353], [0.09816955029964447], [0.295624315738678]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3d37e6e035d79d496e11bf1e91cff7f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3623265326023102], [0.12095697224140167], [0.3224317133426666], [0.4281119704246521]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.13186973333358765], [0.09202171117067337], [0.08312810212373734], [0.2131945937871933]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3caf9a74e26c244968642218b2091139(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.026258420199155807], [0.011381367221474648], [0.030490899458527565], [-0.0003965869836974889]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.06323745846748352], [-0.025388313457369804], [0.017503943294286728], [0.007693326100707054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_dd831fc68a90a7ae09ce4202d408e2d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[3.408273696899414], [3.2306909561157227], [0.42592892050743103], [20.398836135864258]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f866141f6138d4c2131ae7ec085de7de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d76fc14b044ae2fdeb0c7045b2e1ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8e250b5d86a779dd7ce7c899137102c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bbf7c0afdbea438b74c5eae4bcac9b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f64cb6fd9abf54063d21adfc7894a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d76fc14b044ae2fdeb0c7045b2e1ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b64b2d0be6f55b93150ae0072dbd4bc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f9673be4e2e87c1f01dd5143c1840ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5cb451e1752b1d6bd0dd8bfb8c1d248(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_430aa3aa9e88a6965d5a1616f124695e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.059797223657369614, 0.3857690691947937, 0.4381629526615143, 0.1027114987373352], [0.4444314241409302, 0.43357622623443604, 0.13710230588912964, 0.45452260971069336], [0.24993541836738586, 0.26625725626945496, 0.2578131854534149, 0.3478696346282959], [0.24993541836738586, 0.26625725626945496, 0.2578131854534149, 0.3478696346282959], [0.23480947315692902, 0.0843145027756691, 0.08692729473114014, 0.3137372136116028]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.11977646499872208, 0.21320702135562897, 0.14060527086257935, 0.06935633718967438], [0.006631460040807724, 0.2403852343559265, 0.12097999453544617, 0.272177129983902], [0.42431706190109253, 0.021122237667441368, 0.2849246561527252, 0.21280157566070557], [0.42431706190109253, 0.021122237667441368, 0.2849246561527252, 0.21280157566070557], [0.09864906966686249, 0.4189547300338745, 0.3182961344718933, 0.3324736952781677]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f7403d51be6d64480ab264e3e0033ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8113762435adf465fe0daca0d1572755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82878af8922b46863353e8192e9a86b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ebe674d82ace0ab9a0a56ae1122934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d21873e2c3d30761a437e7b43dd6d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6e894825353c335b8bec84889131dcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4deaf6fd6d801a876df9c234a227795
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63ec4cc02e28738a7e86184bcc7e3760(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ebe674d82ace0ab9a0a56ae1122934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75c43c24d6de141762f4371844030d6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.431508332490921, 0.3163807988166809, 0.2813553214073181, 0.42060959339141846], [0.20145507156848907, 0.1395324319601059, 0.32148462533950806, 0.44821107387542725], [0.1956084966659546, 0.00535923708230257, 0.46844279766082764, 0.21772362291812897], [0.431508332490921, 0.3163807988166809, 0.2813553214073181, 0.42060959339141846], [0.48524898290634155, 0.006882299669086933, 0.3400641977787018, 0.2631310820579529], [0.4988814890384674, 0.0466485433280468, 0.2382136881351471, 0.10030604153871536], [0.48524898290634155, 0.006882299669086933, 0.3400641977787018, 0.2631310820579529]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.1268680989742279, 0.3940819203853607, 0.13702614605426788, 0.24026523530483246], [0.24064131081104279, 0.11957230418920517, 0.3499946892261505, 0.48193952441215515], [0.21294118463993073, 0.2877182066440582, 0.11202623695135117, 0.3961676359176636], [0.1268680989742279, 0.3940819203853607, 0.13702614605426788, 0.24026523530483246], [0.42945852875709534, 0.3748156428337097, 0.3677677512168884, 0.013625810854136944], [0.25984281301498413, 0.2521388530731201, 0.14442987740039825, 0.397305965423584], [0.42945852875709534, 0.3748156428337097, 0.3677677512168884, 0.013625810854136944]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ee58507b6beb37b8cc4401cfabb6460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36ee1ec77ca5536bab638d289c99803e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf8043fc38e4ed578df2e49bdfa58027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()