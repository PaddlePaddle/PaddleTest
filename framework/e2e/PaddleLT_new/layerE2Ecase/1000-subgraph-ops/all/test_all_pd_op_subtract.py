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


    class TestPrimitiveOp_d0d91251eac078da8f7df233424fcadc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4618493914604187]], [[0.4058695137500763]], [[0.08289115130901337]], [[0.26278284192085266]], [[0.23262260854244232]], [[0.013348624110221863]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7939244508743286]], [[0.8136937618255615]], [[0.6444892287254333]], [[0.7144414186477661]], [[0.738219141960144]], [[0.6364917755126953]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_1cd96bd0b741453e69e3abc09e97b2db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.23339112102985382]], [[0.22922959923744202]], [[0.20036675035953522]], [[0.05275467038154602]], [[0.09179128706455231]], [[0.08593977987766266]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6195000410079956]], [[0.655858039855957]], [[0.6888777017593384]], [[0.7641112804412842]], [[0.5334721207618713]], [[0.6528226137161255]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_806d72dbfcc3f93f944a2306ff791e7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.29278796911239624, 0.12469929456710815]], [[0.19302858412265778, 0.39783740043640137]], [[0.1291770488023758, 0.45455050468444824]], [[0.012822974473237991, 0.04444433003664017]], [[0.32388806343078613, 0.1391058713197708]], [[0.46277740597724915, 0.001989981159567833]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.42187127470970154, 0.18227285146713257]], [[0.25669822096824646, 0.37202227115631104]], [[0.17325818538665771, 0.39465776085853577]], [[0.18558315932750702, 0.20827585458755493]], [[0.3987077474594116, 0.1856575310230255]], [[0.08304066210985184, 0.22850774228572845]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_8dcdfb700c44c0048d31790fc509e840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.46550989151000977, 0.1697573959827423]], [[0.13966119289398193, 0.4147696793079376]], [[0.2085360884666443, 0.08584899455308914]], [[0.2911050617694855, 0.42420315742492676]], [[0.1459295153617859, 0.45230230689048767]], [[0.01436071377247572, 0.10688220709562302]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.42187127470970154, 0.18227285146713257]], [[0.25669822096824646, 0.37202227115631104]], [[0.17325818538665771, 0.39465776085853577]], [[0.18558315932750702, 0.20827585458755493]], [[0.3987077474594116, 0.1856575310230255]], [[0.08304066210985184, 0.22850774228572845]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_b7ab25ee2cfa78f211dd3fdb91b0a8a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ae748e3ee78da0b91c7817413d3dc0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.4262748956680298, 0.1736283153295517]], [[0.1497272402048111, 0.43034040927886963]], [[0.4598858952522278, 0.04210919141769409]], [[0.2074327915906906, 0.2813599109649658]], [[0.46443691849708557, 0.28645503520965576]], [[0.16248425841331482, 0.24488294124603271]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_ce4b3f83834348c3a5df8c24a6b3fb88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.4765012264251709, 0.2170112580060959, 0.4702689051628113, 0.4369574785232544, 0.244343563914299, 0.49530693888664246, 0.3121092915534973, 0.0963035523891449, 0.38352274894714355, 0.009155333042144775, 0.4616371691226959, 0.020641636103391647, 0.15587159991264343, 0.02491425909101963, 0.443154513835907, 0.24451406300067902], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_9cb18e4462f220e11b7fa62202ec5a26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4765012264251709, 0.2170112580060959, 0.4702689051628113, 0.4369574785232544, 0.244343563914299, 0.49530693888664246, 0.3121092915534973, 0.0963035523891449, 0.38352274894714355, 0.009155333042144775, 0.4616371691226959, 0.020641636103391647, 0.15587159991264343, 0.02491425909101963, 0.443154513835907, 0.24451406300067902], dtype='float32').reshape([16]),
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


    class TestPrimitiveOp_90121b319cead1566421c49e41b71afc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9959affea9b84830410605c58917f22a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_90121b319cead1566421c49e41b71afc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d46df5ec11068e5d2ebe6414441ec4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.45848238468170166, 0.34971383213996887, 0.3945930004119873, 0.447701632976532], [0.41079744696617126, 0.0009267344721592963, 0.08779380470514297, 0.06212601438164711], [0.2339312881231308, 0.2007937729358673, 0.15278489887714386, 0.3617117404937744], [0.4200783371925354, 0.41086649894714355, 0.39261895418167114, 0.24603603780269623], [0.32407864928245544, 0.22759559750556946, 0.45276063680648804, 0.4318024814128876]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.4423430263996124, 0.3251437842845917, 0.2146957665681839, 0.06303687393665314], [0.09525219351053238, 0.11553490161895752, 0.0325784906744957, 0.13175402581691742], [0.2793131172657013, 0.4104779064655304, 0.26631882786750793, 0.19468954205513], [0.10130643099546432, 0.19705356657505035, 0.33797094225883484, 0.3692222237586975], [0.18079429864883423, 0.10179086029529572, 0.4477527141571045, 0.4130363166332245]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_18d12dfe556373bfed233390542ddf09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015340379439294338, 0.024809498339891434, 0.4150008261203766, 0.38157200813293457], [0.4599052965641022, 0.14341849088668823, 0.12601007521152496, 0.34754735231399536], [0.49729982018470764, 0.24716007709503174, 0.34805798530578613, 0.10370808094739914], [0.4599052965641022, 0.14341849088668823, 0.12601007521152496, 0.34754735231399536], [0.49729982018470764, 0.24716007709503174, 0.34805798530578613, 0.10370808094739914]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.18711674213409424, 0.19057035446166992, 0.2047814428806305, 0.4759475886821747], [0.09319417178630829, 0.05631349980831146, 0.031228844076395035, 0.19341900944709778], [0.08189824223518372, 0.08736616373062134, 0.3658214509487152, 0.29241862893104553], [0.09319417178630829, 0.05631349980831146, 0.031228844076395035, 0.19341900944709778], [0.08189824223518372, 0.08736616373062134, 0.3658214509487152, 0.29241862893104553]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_712e1be40545814c3fca2aa24262ae9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.2974533438682556], [0.04537849500775337], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_b7038710d38ad77bf74f4448a686ad47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09990614652633667], [0.2677699625492096], [0.015402782708406448], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.1261843889951706], [0.030494073405861855], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.04978129267692566], [0.2015986442565918], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.3369775414466858], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_4bbde2670142f95da11786eb93148a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.2974533438682556], [0.04537849500775337], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e99402103a964419db7cc66e3e3bf0e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.2677699625492096], [0.015402782708406448], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.1261843889951706], [0.05905040726065636], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.04978129267692566], [0.042689915746450424], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.11604061722755432], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_a0e6a022bb7c5df5826a182ea11c228d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.4127102792263031], [0.41945791244506836], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_be878a040d4295576e8eae326a802ac8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09990614652633667], [0.32696646451950073], [0.2620046138763428], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.29109078645706177], [0.030494073405861855], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.0034736385568976402], [0.2015986442565918], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.3369775414466858], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2676e1f6d07268be9c7acd4fdd4c7967(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05231598764657974], [0.009299254976212978], [0.10837691277265549], [-0.07017733156681061], [0.08777499198913574], [-0.016110287979245186], [-0.0599064826965332], [0.08122386783361435], [-0.026601403951644897]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.010175604373216629], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_da08024382ac218db477bc6350f59684(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.4127102792263031], [0.41945791244506836], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_d571809b30dca09539b5ba28c8160331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.32696646451950073], [0.2620046138763428], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.29109078645706177], [0.05905040726065636], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.0034736385568976402], [0.042689915746450424], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.11604061722755432], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_715886cc893a6fd3b4f71859b5031113(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05092363804578781], [0.038372911512851715], [0.10456927120685577], [0.090772345662117], [0.09837325662374496], [0.010857976041734219], [-0.014304488897323608], [-0.004792569670826197], [0.08330294489860535]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.05231598764657974], [0.009299254976212978], [0.10837691277265549], [-0.07017733156681061], [0.07759939134120941], [-0.016110287979245186], [-0.0599064826965332], [0.08122386783361435], [-0.026601403951644897]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e63d7544515b5854633cde355353a35f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [-0.0], [0.13112995028495789], [-0.0], [-0.0], [0.0], [-0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.02734191156923771], [0.7576609253883362], [-0.03641262277960777], [1.7731136083602905], [0.2111739069223404], [2.4837284088134766], [-3.1879498958587646], [17.947874069213867], [1.3193333148956299]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_1af58d553c5e7b6ec645021f13cf8a4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5822a565ae043a4fa59022614565406(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.21194247901439667]], [[0.3961002826690674]], [[0.4367770254611969]], [[0.30254191160202026]], [[0.17525893449783325]], [[0.3245100975036621]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6667161583900452]], [[0.8103682398796082]], [[0.75272536277771]], [[0.6059465408325195]], [[0.7161102294921875]], [[0.8132511377334595]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_a0de73b450c16800ce9c578aeb02393c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.12078697234392166]], [[0.08913929015398026]], [[0.149296835064888]], [[0.39876019954681396]], [[0.13448122143745422]], [[0.1268261820077896]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.5343883037567139]], [[0.513643741607666]], [[0.5729232430458069]], [[0.5704134106636047]], [[0.7160937786102295]], [[0.7558856010437012]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_c251930a31003e9c58dde854cb5874d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73161c186116e7297fd5e8870ca5327b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c251930a31003e9c58dde854cb5874d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e814782cf8084e98910d1afb21c39abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3001679480075836, 0.03978395089507103, 0.13419044017791748, 0.09352608025074005], [0.39127877354621887, 0.4333009123802185, 0.06490619480609894, 0.38775861263275146], [0.39391446113586426, 0.4543362259864807, 0.011423652060329914, 0.07635422796010971], [0.39127877354621887, 0.4333009123802185, 0.06490619480609894, 0.38775861263275146], [0.39391446113586426, 0.4543362259864807, 0.011423652060329914, 0.07635422796010971], [0.4616011083126068, 0.20268231630325317, 0.38833072781562805, 0.20092904567718506], [0.4616011083126068, 0.20268231630325317, 0.38833072781562805, 0.20092904567718506]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.295341819524765, 0.11133721470832825, 0.12058486044406891, 0.14470656216144562], [0.2465321570634842, 0.16365154087543488, 0.0819966197013855, 0.021441368386149406], [0.0940665453672409, 0.44933849573135376, 0.40764477849006653, 0.22932206094264984], [0.2465321570634842, 0.16365154087543488, 0.0819966197013855, 0.021441368386149406], [0.0940665453672409, 0.44933849573135376, 0.40764477849006653, 0.22932206094264984], [0.23235131800174713, 0.08092588931322098, 0.4600076675415039, 0.27263739705085754], [0.23235131800174713, 0.08092588931322098, 0.4600076675415039, 0.27263739705085754]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_b11dbe1bc6ebf3a37e473d7536553d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.33742555975914, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.09696357697248459], dtype='float32').reshape([6]),
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c4c69ccf0263e70e002a9d67a4eed8c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.2536962032318115, 0.2332039326429367, 0.22083275020122528, 0.2845509946346283, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8fb6832185ca70dee28293d7ddf5f8cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.07290997356176376, 0.22891433537006378, 0.06531837582588196, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_9fab32bc0be449931b7f7343dbac446f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2081550508737564, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.285847932100296], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.10902168601751328, 0.04308721795678139, 0.134864941239357, 0.4397190809249878, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ad6864c8b87a8d9898121090f34a3d4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.07290997356176376, 0.22891433537006378, 0.223669171333313, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e1d2fd63be576e1d50a05983861bd7fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.4702686071395874, 0.36698785424232483, 0.134864941239357, 0.4465831220149994, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8d8cb710c6b9af296a20d8014b8b5240(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8303440e5665b5050f220e36b8ca9cca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0f6edb6a2b08fc91d6acc22d6de7a1d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.020324425771832466, 0.013810954988002777, -0.004959273152053356, 0.040052562952041626, -0.04735404625535011, -0.022230884060263634], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ad20ab878ccb3247d8f11388f309854c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.35703232884407043, 0.2805730104446411, 0.1354352831840515, 0.15389223396778107, 0.2500401735305786, 0.16624096035957336], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2881614863872528, 0.071312814950943, 0.06014230102300644, 0.2271801382303238, 0.12440825253725052, 0.28540241718292236], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e0f87e8fb8a595ae8b5c46914fcad387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.17309994995594025, 0.36198240518569946, 0.30009588599205017, 0.1426411271095276, 0.36556705832481384, 0.08817589282989502], dtype='float32').reshape([6]),
                paddle.to_tensor([0.26979461312294006, 0.2617741525173187, 0.14019326865673065, 0.12376508861780167, 0.23937153816223145, 0.2517835199832916], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_74ac441ef9838f0602f654f4679f615b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.040052223950624466, 0.026079408824443817, 0.06531837582588196, 0.23551833629608154], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_017cd302cd3dbebf3341972ad2d2f4c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2081550508737564, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.285847932100296], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1651548445224762, 0.10902168601751328, 0.04308721795678139, 0.0644494965672493, 0.4397190809249878, 0.07492171972990036], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ec003c842b6388bd0b10a3955bcb761c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.9287320971488953, 0.14690853655338287, -0.13073183596134186, 0.15498313307762146, -0.28680527210235596, -1.3649654388427734], dtype='float32').reshape([6]),
                paddle.to_tensor([-1.1857959032058716, -1.0462806224822998, -0.9591996669769287, 1.0217697620391846, -0.31468695402145386, -1.3817603588104248], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_151f323381f1641a3cabb1f80166cb45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0cb9e979fc0902ef8025a93541cba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_151f323381f1641a3cabb1f80166cb45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8265d2b3756980f2d194236a95721671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.3306543529033661, 0.3834773600101471, 0.49611005187034607, 0.11812765151262283, 0.478644460439682, 0.32757896184921265, 0.3514584004878998, 0.3204523026943207, 0.48325031995773315, 0.25242266058921814, 0.4594441056251526, 0.10603760182857513, 0.09884601831436157, 0.039876293390989304, 0.12443646788597107, 0.20525671541690826, 0.031096970662474632, 0.08158884942531586, 0.2653093934059143, 0.3785844147205353, 0.4728078544139862, 0.47888636589050293, 0.37857428193092346, 0.030704200267791748], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_fa7d058ea1700290211a447a8442a35f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3306543529033661, 0.3834773600101471, 0.49611005187034607, 0.11812765151262283, 0.478644460439682, 0.32757896184921265, 0.3514584004878998, 0.3204523026943207, 0.48325031995773315, 0.25242266058921814, 0.4594441056251526, 0.10603760182857513, 0.09884601831436157, 0.039876293390989304, 0.12443646788597107, 0.20525671541690826, 0.031096970662474632, 0.08158884942531586, 0.2653093934059143, 0.3785844147205353, 0.4728078544139862, 0.47888636589050293, 0.37857428193092346, 0.030704200267791748], dtype='float32').reshape([24]),
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


    class TestPrimitiveOp_8be27c6414a5c8e9a1948dafb5ba5640(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_287688f7ae6e37c2608a6e417062fdcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8be27c6414a5c8e9a1948dafb5ba5640(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_1c1f31afa1d7619193bdc80164594255(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.004606406204402447, 0.23123939335346222, 0.4518365263938904, 0.139329195022583], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_0a1c7c9df8dddeb4b34e5c991ea9f374(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.004606406204402447, 0.23123939335346222, 0.4518365263938904, 0.139329195022583], dtype='float32').reshape([4]),
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


    class TestPrimitiveOp_4c37a7b9139b876a4a2f0d831b63f650(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3088763952255249, 0.12988963723182678, 0.4854060709476471, 0.29282501339912415], [0.15517044067382812, 0.20120646059513092, 0.17029418051242828, 0.21977749466896057], [0.40536436438560486, 0.4242015480995178, 0.1562468409538269, 0.3942132294178009], [0.4052164852619171, 0.16592980921268463, 0.16075582802295685, 0.11449424177408218], [0.4052164852619171, 0.16592980921268463, 0.16075582802295685, 0.11449424177408218], [0.40536436438560486, 0.4242015480995178, 0.1562468409538269, 0.3942132294178009]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.3819200098514557, 0.23831237852573395, 0.25062572956085205, 0.26940813660621643], [0.11866627633571625, 0.4657531976699829, 0.4957130253314972, 0.3083881437778473], [0.1402580738067627, 0.3681163191795349, 0.37202703952789307, 0.2937219440937042], [0.327421098947525, 0.23211318254470825, 0.1837078183889389, 0.3576776087284088], [0.327421098947525, 0.23211318254470825, 0.1837078183889389, 0.3576776087284088], [0.1402580738067627, 0.3681163191795349, 0.37202703952789307, 0.2937219440937042]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_f8382f69cba3292720aa15e70db2a675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4678221344947815, 0.04469945654273033, 0.23884932696819305, 0.25630196928977966], [0.06454280763864517, 0.35123002529144287, 0.2817023694515228, 0.07664187997579575], [0.387455552816391, 0.08136474341154099, 0.36431193351745605, 0.0010941538494080305], [0.13462622463703156, 0.4734646677970886, 0.328313946723938, 0.43288931250572205], [0.4678221344947815, 0.04469945654273033, 0.23884932696819305, 0.25630196928977966]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.4540683627128601, 0.33153462409973145, 0.20022638142108917, 0.06467278301715851], [0.44031772017478943, 0.4055336117744446, 0.034483954310417175, 0.1595418006181717], [0.2573700547218323, 0.22866977751255035, 0.2128351330757141, 0.3317737579345703], [0.386608362197876, 0.10989176481962204, 0.45954629778862, 0.22013701498508453], [0.4540683627128601, 0.33153462409973145, 0.20022638142108917, 0.06467278301715851]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_73ece6d168a08bd3fa887a7c95aa72af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bee3f060aceabee63629beb3f0dccf5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9c32dfd5a108b6241abc2e187ce297d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_322ae80d88857285e2afe610cff5c342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b0e6a49df85f5cc66f6606b92ee22a05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1846d81dda1f23fee4ee75e86e124ebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9c32dfd5a108b6241abc2e187ce297d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c54ce2c76fb2d05f1bb005484a3f464f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0011826035333797336]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_a7bf542e908842b46d8a5864c04fdd8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b0e6a49df85f5cc66f6606b92ee22a05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8e07ae180a1a2c837d0ef31bf19d42ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03425128385424614]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0011826036497950554]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_5236ea01db42b87f4d7210deb87c22c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.9654726982116699]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_841d1f35a585136b954a538ebd908f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.14439257979393005], [0.09650922566652298], [0.16213607788085938], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3966229259967804], [0.4306377172470093], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_92395d4ac97f39198c55d4cba1e9aeab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_b0738fdb03fefeae5e31e71b827d2ff0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.3532122075557709], [0.09650922566652298], [0.2863064706325531], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3966229259967804], [0.2040838748216629], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ff82bf91aa7fc5faa224a0ff4fa1e111(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_cc6e32649afca92bfce4b82071455340(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4361984133720398], [0.14439257979393005], [0.49514085054397583], [0.16213607788085938], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.4306377172470093], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d33ebc43fb70346fdeb41db34e1baac2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e2a800767c8c145c5bc88710bbdbf19e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24575075507164001], [-0.04205697402358055], [0.0034978860057890415], [0.003514286130666733], [0.16795028746128082], [0.1216726154088974]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.014378788881003857], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_23948f1b1b43f4396d9c6278e5a7c8e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4361984133720398], [0.3532122075557709], [0.49514085054397583], [0.2863064706325531], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.2040838748216629], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_6db3dec6f7c0d352c90b9dc4779ca4a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d2e0ed2cf6488071195df7349be43fa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09241148084402084], [0.023195823654532433], [-0.002818223787471652], [0.026250513270497322], [0.1769580990076065], [0.013878539204597473]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.24575075507164001], [-0.04205697402358055], [0.0034978860057890415], [0.003514286130666733], [0.1535715013742447], [0.1216726154088974]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_40bfc2224a371d51189d33d508137c5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [0.0], [0.0], [0.09362927824258804], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-1.6593097448349], [2.813126802444458], [2.241166830062866], [0.8661250472068787], [0.13215894997119904], [-7.766961097717285]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ddb143a473b351c891d3af24a72c8b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20391754806041718, 0.31501132249832153, 0.07210913300514221, 0.2850325405597687], [0.42501765489578247, 0.20123079419136047, 0.21216268837451935, 0.4939834475517273], [0.10778137296438217, 0.2493751496076584, 0.11860713362693787, 0.3470500409603119], [0.18004286289215088, 0.1343468427658081, 0.10950658470392227, 0.3252163529396057]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.10165496915578842, 0.17966170608997345, 0.15234315395355225, 0.2450416386127472], [0.12599779665470123, 0.2939036786556244, 0.42129790782928467, 0.0379941388964653], [0.30939817428588867, 0.3820403516292572, 0.22087596356868744, 0.14539435505867004], [0.17156982421875, 0.10984878242015839, 0.1759537160396576, 0.06420376151800156]], dtype='float32').reshape([4, 4]),
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


    class TestPrimitiveOp_5a9812e26d44a8a4d0140bd3f2bba2e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732f6967765247392fe0f05a5aa3a9c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_5a9812e26d44a8a4d0140bd3f2bba2e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67328a2fbcb4caf2961f75be9a29808f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17218145728111267, 0.06576531380414963, 0.08751874417066574, 0.30011749267578125], [0.17218145728111267, 0.06576531380414963, 0.08751874417066574, 0.30011749267578125], [0.20134691894054413, 0.11920982599258423, 0.2738986015319824, 0.25143593549728394], [0.17379897832870483, 0.22546128928661346, 0.18680396676063538, 0.008240723982453346], [0.09319761395454407, 0.1083202138543129, 0.17002469301223755, 0.3804638981819153], [0.1670711785554886, 0.051270559430122375, 0.39242714643478394, 0.3678208291530609], [0.07703103125095367, 0.42867806553840637, 0.35202184319496155, 0.48333871364593506]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.014671501703560352, 0.33271241188049316, 0.4347265660762787, 0.0826963558793068], [0.014671501703560352, 0.33271241188049316, 0.4347265660762787, 0.0826963558793068], [0.050411611795425415, 0.2639580965042114, 0.4960781931877136, 0.15688583254814148], [0.04975507780909538, 0.2948163151741028, 0.31037741899490356, 0.3716491460800171], [0.07919828593730927, 0.49961045384407043, 0.35128462314605713, 0.17356255650520325], [0.1732870638370514, 0.1521768718957901, 0.08007995039224625, 0.22500772774219513], [0.4506492614746094, 0.35124221444129944, 0.08421099931001663, 0.22466899454593658]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_976f5c10d07168303f727dfec20cbfb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7a66bc3949f8370898008d3b8e7dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_976f5c10d07168303f727dfec20cbfb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbe9ee7ed1efef631c176e4de428daba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac39268a14eada43b38d17a103f32ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_fbe9ee7ed1efef631c176e4de428daba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a89dfe01d48a19b39a14b931a6f22cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_968be798066db33bf4ba449d629257c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20043645799160004, 0.4294639527797699, 0.2814168930053711, 0.1474073827266693], [0.046092040836811066, 0.22925116121768951, 0.28526708483695984, 0.15180309116840363], [0.046092040836811066, 0.22925116121768951, 0.28526708483695984, 0.15180309116840363], [0.4408273696899414, 0.30982300639152527, 0.4029184579849243, 0.18092027306556702], [0.2481885403394699, 0.33687835931777954, 0.37247517704963684, 0.227286696434021], [0.4674718976020813, 0.22988873720169067, 0.3623161315917969, 0.3304578959941864]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.07959121465682983, 0.2769489586353302, 0.46637436747550964, 0.364219069480896], [0.03007488325238228, 0.23571008443832397, 0.4971219003200531, 0.3081081807613373], [0.03007488325238228, 0.23571008443832397, 0.4971219003200531, 0.3081081807613373], [0.08231888711452484, 0.19332893192768097, 0.23069065809249878, 0.4883212447166443], [0.22639085352420807, 0.29252418875694275, 0.15105053782463074, 0.4269372522830963], [0.44897162914276123, 0.1686171293258667, 0.49535754323005676, 0.23849567770957947]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_7de7012a311860dd715448fe8b0f2cd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f62c64312ccb8d87c12c185fd5e515e
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.06011958047747612, 27.418258666992188, 2.888007402420044, 1.416237473487854], [0.17004764080047607, 1.3946164846420288, 0.7928432822227478, 0.07617802172899246]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_b4546185d8a8c1c05607fc1df89667e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3db6dde5bf3f37813420aec60ad447b6
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1.1455035209655762, 0.9406334161758423, 0.9402686357498169, 2.975639820098877], [9.551335334777832, 1.0109132528305054, 2.315927267074585, 0.2375415414571762]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_f27f94c3ae1973a2e8432dd80a318111(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.3282714784145355], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e5c9786c42f95b676e1ecf861f106cab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.31906890869140625], [0.04625708982348442], [0.439250111579895], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_1ec8786a9b522fabfba0c589acc22c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.36288416385650635], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9b1ab8f3ba30ad25e08c7ec94d2d5608(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.31906890869140625], [0.02482573315501213], [0.439250111579895], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_cbf3f029c7bebc4af2860099508d2b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.3282714784145355], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f6e412b64b1321370f360befb81fe668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.04625708982348442], [0.1838442087173462], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_409056cbe015232ad1fddd5e03416095(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.005169573239982128], [0.06563594937324524], [0.019044138491153717], [0.1293209046125412], [-0.01654825359582901]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_ad2aadbe9531128fa3d9d06b5c4e031b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.36288416385650635], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e5e82d6649e2aa81e24416ca650f8ceb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.02482573315501213], [0.1838442087173462], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_8ecddb585842466f44a022f10ea533f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.012724664062261581], [0.08693233877420425], [0.11657001823186874], [0.09260812401771545], [0.010530282743275166]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.005169573239982128], [0.06563594937324524], [0.019044138491153717], [0.1293209046125412], [-0.01654825359582901]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_3289c1cdb1e147626c084ecff5033a5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[1.406264066696167], [0.24497660994529724], [0.8366292119026184], [-0.3964315354824066], [2.5714917182922363]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_a4eb660eebac99c85997677da75f41d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45c695e42a6813c9472c726bd11547cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a4eb660eebac99c85997677da75f41d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95a0267f24b2f1e5dca4b772413c2b8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28f1164875da24f64952727e92f1f18f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_95a0267f24b2f1e5dca4b772413c2b8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bccc51fed13c06f84d7bdc548579dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c21bd6bece4ddc38b84e0370da9eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7bccc51fed13c06f84d7bdc548579dc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_da573b992cb67d478a1f365d9f90c432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.34141385555267334, 0.08070738613605499, 0.03860168904066086, 0.49608147144317627, 0.41707998514175415, 0.07405710965394974, 0.39049777388572693, 0.14312376081943512, 0.4575801193714142, 0.44072431325912476, 0.48256993293762207, 0.10408809781074524, 0.3957173824310303, 0.045056845992803574, 0.2210846096277237, 0.11286043375730515, 0.109720878303051, 0.17332060635089874, 0.47122129797935486, 0.48317795991897583], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_babe53a7f60a08fa112856b8eab6ec44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.34141385555267334, 0.08070738613605499, 0.03860168904066086, 0.49608147144317627, 0.41707998514175415, 0.07405710965394974, 0.39049777388572693, 0.14312376081943512, 0.4575801193714142, 0.44072431325912476, 0.48256993293762207, 0.10408809781074524, 0.3957173824310303, 0.045056845992803574, 0.2210846096277237, 0.11286043375730515, 0.109720878303051, 0.17332060635089874, 0.47122129797935486, 0.48317795991897583], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_ceb34b6ab5cd4c0956d4fd4f79209f6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14522317051887512], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_bc9b89520247c368be95c21d5048165a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.440621554851532], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.32093942165374756], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7fc4d437235ad0c98453977dff3950cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_0f4413c5b528c6956fc773e5c0432c13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.48723578453063965], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.32093942165374756], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_c3997395517807b014087c67224b8aee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14522317051887512], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_8c284825fd4168f064b8502e78587e4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28212907910346985], [0.440621554851532], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_2260b4770e4a96c82251955f2a0f1d6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.04221170023083687], [0.1649954915046692], [-0.0011601648293435574], [-0.019290968775749207]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.02299167960882187], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_0bd096cf33b73ba8b7fee739793e38ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_cf4680715055edbf1d218d60ba8f1c29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28212907910346985], [0.48723578453063965], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_62e79b6ec3762afef49d29166928ca15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.011106634512543678], [0.16120393574237823], [0.0076672681607306], [0.06769919395446777]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.04221170023083687], [0.14200380444526672], [-0.0011601647129282355], [-0.019290968775749207]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_dcaf0837bc8e85995d5b0979e349a080(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.16190889477729797], [-0.0], [-0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[4.800584316253662], [0.11910460889339447], [1.1513140201568604], [1.2849512100219727]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f866141f6138d4c2131ae7ec085de7de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ad856c68b508647f28e4a711dbb8f2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b4d34acd377030a455506a6e590be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4ad856c68b508647f28e4a711dbb8f2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a4111e1475b599fa44744ebe9879d21b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18006691336631775, 0.47249430418014526, 0.2704814374446869, 0.28356489539146423], [0.1444801688194275, 0.26884371042251587, 0.32883790135383606, 0.40756726264953613], [0.40327703952789307, 0.16330288350582123, 0.4570107161998749, 0.3071627914905548], [0.40327703952789307, 0.16330288350582123, 0.4570107161998749, 0.3071627914905548], [0.4027557671070099, 0.4091954827308655, 0.3253519833087921, 0.04934440553188324]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.18216904997825623, 0.3664987087249756, 0.09633531421422958, 0.16741898655891418], [0.432188481092453, 0.08552742004394531, 0.4801865220069885, 0.10212612897157669], [0.09275709837675095, 0.3269622027873993, 0.2648838758468628, 0.37689346075057983], [0.09275709837675095, 0.3269622027873993, 0.2648838758468628, 0.37689346075057983], [0.1461302638053894, 0.10133280605077744, 0.3515809178352356, 0.4469776153564453]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_ecdb97e984373b83ddaaed3951d16d23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6f6d5d7fbe1a2f0ed6eb7a825e84954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_ecdb97e984373b83ddaaed3951d16d23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d24d2e299067f0114ab676603b7437a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07471240311861038, 0.1784989833831787, 0.0961991548538208, 0.2725340723991394], [0.4185662865638733, 0.41786858439445496, 0.3821410536766052, 0.3345850706100464], [0.39434146881103516, 0.07983526587486267, 0.014400581829249859, 0.29254430532455444], [0.07471240311861038, 0.1784989833831787, 0.0961991548538208, 0.2725340723991394], [0.348670095205307, 0.276864618062973, 0.2532390356063843, 0.433200865983963], [0.07939526438713074, 0.25467807054519653, 0.49734240770339966, 0.2597086429595947], [0.348670095205307, 0.276864618062973, 0.2532390356063843, 0.433200865983963]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.1009109765291214, 0.05554174259305, 0.07083551585674286, 0.32371416687965393], [0.15809713304042816, 0.25184744596481323, 0.13013799488544464, 0.4204864799976349], [0.12200950086116791, 0.45094767212867737, 0.051762133836746216, 0.018873345106840134], [0.1009109765291214, 0.05554174259305, 0.07083551585674286, 0.32371416687965393], [0.3411698639392853, 0.017546646296977997, 0.26674339175224304, 0.05254773423075676], [0.46994319558143616, 0.45695093274116516, 0.27365565299987793, 0.011935419403016567], [0.3411698639392853, 0.017546646296977997, 0.26674339175224304, 0.05254773423075676]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_4efa398138aaa81aa51e73798bab7d74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4618493914604187]], [[0.4058695137500763]], [[0.08289115130901337]], [[0.26278284192085266]], [[0.23262260854244232]], [[0.013348624110221863]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7939244508743286]], [[0.8136937618255615]], [[0.6444892287254333]], [[0.7144414186477661]], [[0.738219141960144]], [[0.6364917755126953]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_a52991e4eab00a59855b664d6697ccfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.23339112102985382]], [[0.22922959923744202]], [[0.20036675035953522]], [[0.05275467038154602]], [[0.09179128706455231]], [[0.08593977987766266]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6195000410079956]], [[0.655858039855957]], [[0.6888777017593384]], [[0.7641112804412842]], [[0.5334721207618713]], [[0.6528226137161255]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_ba9190f45d6c16f68f80ac4c3203424b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_758bf3801ed88050004599925b40e60d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.29278796911239624, 0.12469929456710815]], [[0.19302858412265778, 0.39783740043640137]], [[0.1291770488023758, 0.45455050468444824]], [[0.012822974473237991, 0.04444433003664017]], [[0.32388806343078613, 0.1391058713197708]], [[0.46277740597724915, 0.001989981159567833]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.42187127470970154, 0.18227285146713257]], [[0.25669822096824646, 0.37202227115631104]], [[0.17325818538665771, 0.39465776085853577]], [[0.18558315932750702, 0.20827585458755493]], [[0.3987077474594116, 0.1856575310230255]], [[0.08304066210985184, 0.22850774228572845]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_dab9f31b6df47b185bcd72261033d646(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_758bf3801ed88050004599925b40e60d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.46550989151000977, 0.1697573959827423]], [[0.13966119289398193, 0.4147696793079376]], [[0.2085360884666443, 0.08584899455308914]], [[0.2911050617694855, 0.42420315742492676]], [[0.1459295153617859, 0.45230230689048767]], [[0.01436071377247572, 0.10688220709562302]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.42187127470970154, 0.18227285146713257]], [[0.25669822096824646, 0.37202227115631104]], [[0.17325818538665771, 0.39465776085853577]], [[0.18558315932750702, 0.20827585458755493]], [[0.3987077474594116, 0.1856575310230255]], [[0.08304066210985184, 0.22850774228572845]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_6352c64a7c6032f3ff549c9165e9b264(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e04add1b8d6435475bb578528599f0c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.4262748956680298, 0.1736283153295517]], [[0.1497272402048111, 0.43034040927886963]], [[0.4598858952522278, 0.04210919141769409]], [[0.2074327915906906, 0.2813599109649658]], [[0.46443691849708557, 0.28645503520965576]], [[0.16248425841331482, 0.24488294124603271]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_b4503c28217201608271a91c722857a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ec4a74d6c67bc08e22f55b7a09eba1e
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.4765012264251709, 0.2170112580060959, 0.4702689051628113, 0.4369574785232544, 0.244343563914299, 0.49530693888664246, 0.3121092915534973, 0.0963035523891449, 0.38352274894714355, 0.009155333042144775, 0.4616371691226959, 0.020641636103391647, 0.15587159991264343, 0.02491425909101963, 0.443154513835907, 0.24451406300067902], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_579c1470a96beff27d22c7773d5c8c8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ec4a74d6c67bc08e22f55b7a09eba1e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4765012264251709, 0.2170112580060959, 0.4702689051628113, 0.4369574785232544, 0.244343563914299, 0.49530693888664246, 0.3121092915534973, 0.0963035523891449, 0.38352274894714355, 0.009155333042144775, 0.4616371691226959, 0.020641636103391647, 0.15587159991264343, 0.02491425909101963, 0.443154513835907, 0.24451406300067902], dtype='float32').reshape([16]),
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


    
    class PrimitiveOp_861ccc41b8aea96d09e2904465ed48ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1723, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dcd686685f08e7e25b9e12f832aaa092(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_861ccc41b8aea96d09e2904465ed48ea
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_51ca59014f76bd8deffb518b4e16f185(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88d119cc27d7f3672afed849a030e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ca59014f76bd8deffb518b4e16f185
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_dcd686685f08e7e25b9e12f832aaa092(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_861ccc41b8aea96d09e2904465ed48ea
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_263f627d48a4424d50bf3acf558db673(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.45848238468170166, 0.34971383213996887, 0.3945930004119873, 0.447701632976532], [0.41079744696617126, 0.0009267344721592963, 0.08779380470514297, 0.06212601438164711], [0.2339312881231308, 0.2007937729358673, 0.15278489887714386, 0.3617117404937744], [0.4200783371925354, 0.41086649894714355, 0.39261895418167114, 0.24603603780269623], [0.32407864928245544, 0.22759559750556946, 0.45276063680648804, 0.4318024814128876]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.4423430263996124, 0.3251437842845917, 0.2146957665681839, 0.06303687393665314], [0.09525219351053238, 0.11553490161895752, 0.0325784906744957, 0.13175402581691742], [0.2793131172657013, 0.4104779064655304, 0.26631882786750793, 0.19468954205513], [0.10130643099546432, 0.19705356657505035, 0.33797094225883484, 0.3692222237586975], [0.18079429864883423, 0.10179086029529572, 0.4477527141571045, 0.4130363166332245]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_ffd624de906ced2b959df993abd6accf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015340379439294338, 0.024809498339891434, 0.4150008261203766, 0.38157200813293457], [0.4599052965641022, 0.14341849088668823, 0.12601007521152496, 0.34754735231399536], [0.49729982018470764, 0.24716007709503174, 0.34805798530578613, 0.10370808094739914], [0.4599052965641022, 0.14341849088668823, 0.12601007521152496, 0.34754735231399536], [0.49729982018470764, 0.24716007709503174, 0.34805798530578613, 0.10370808094739914]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.18711674213409424, 0.19057035446166992, 0.2047814428806305, 0.4759475886821747], [0.09319417178630829, 0.05631349980831146, 0.031228844076395035, 0.19341900944709778], [0.08189824223518372, 0.08736616373062134, 0.3658214509487152, 0.29241862893104553], [0.09319417178630829, 0.05631349980831146, 0.031228844076395035, 0.19341900944709778], [0.08189824223518372, 0.08736616373062134, 0.3658214509487152, 0.29241862893104553]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_93d60c17882555fbca1ee792122a1d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.2974533438682556], [0.04537849500775337], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_3c966ea25245fa0fab367f3bac9acb5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09990614652633667], [0.2677699625492096], [0.015402782708406448], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.1261843889951706], [0.030494073405861855], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.04978129267692566], [0.2015986442565918], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.3369775414466858], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_5baad3fe5e78d7b9387c2449ca263989(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.2974533438682556], [0.04537849500775337], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_0544abf6cf1bffbce8538563ee07e571(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.2677699625492096], [0.015402782708406448], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.1261843889951706], [0.05905040726065636], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.04978129267692566], [0.042689915746450424], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.11604061722755432], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_457c482301c0d60e68953e2da3fa80a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.4127102792263031], [0.41945791244506836], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_7a533eca50567bb4c3627a9ffce09a52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09990614652633667], [0.32696646451950073], [0.2620046138763428], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.29109078645706177], [0.030494073405861855], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.0034736385568976402], [0.2015986442565918], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.3369775414466858], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_a3211d00e472c16ed5592a1af1cb1585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05231598764657974], [0.009299254976212978], [0.10837691277265549], [-0.07017733156681061], [0.08777499198913574], [-0.016110287979245186], [-0.0599064826965332], [0.08122386783361435], [-0.026601403951644897]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.010175604373216629], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_3e2aa987d8539e1d90ecc886c55ef42e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.4127102792263031], [0.41945791244506836], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_066dd362fc8d440b20ba8d1f7ee3ea5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.32696646451950073], [0.2620046138763428], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.29109078645706177], [0.05905040726065636], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.0034736385568976402], [0.042689915746450424], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.11604061722755432], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_1d1b9757f5f5e7822270e4add8f83ab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05092363804578781], [0.038372911512851715], [0.10456927120685577], [0.090772345662117], [0.09837325662374496], [0.010857976041734219], [-0.014304488897323608], [-0.004792569670826197], [0.08330294489860535]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.05231598764657974], [0.009299254976212978], [0.10837691277265549], [-0.07017733156681061], [0.07759939134120941], [-0.016110287979245186], [-0.0599064826965332], [0.08122386783361435], [-0.026601403951644897]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_655a2a20e92c00ba926c39b210099b4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [-0.0], [0.13112995028495789], [-0.0], [-0.0], [0.0], [-0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.02734191156923771], [0.7576609253883362], [-0.03641262277960777], [1.7731136083602905], [0.2111739069223404], [2.4837284088134766], [-3.1879498958587646], [17.947874069213867], [1.3193333148956299]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_c99cd3f9b9c6703e57a2a485e75cd922(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.21194247901439667]], [[0.3961002826690674]], [[0.4367770254611969]], [[0.30254191160202026]], [[0.17525893449783325]], [[0.3245100975036621]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6667161583900452]], [[0.8103682398796082]], [[0.75272536277771]], [[0.6059465408325195]], [[0.7161102294921875]], [[0.8132511377334595]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_2948f72ee80cfbe08d307d8067c77b8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.12078697234392166]], [[0.08913929015398026]], [[0.149296835064888]], [[0.39876019954681396]], [[0.13448122143745422]], [[0.1268261820077896]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.5343883037567139]], [[0.513643741607666]], [[0.5729232430458069]], [[0.5704134106636047]], [[0.7160937786102295]], [[0.7558856010437012]]], dtype='float32').reshape([6, 1, 1]),
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


    
    class PrimitiveOp_58efd46565f914100614802e53be6748(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[5498, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_314f7f82afeeb12c1301383bfe6c6bc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58efd46565f914100614802e53be6748
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9390adba491fe1d012a10b16c533298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e60096827e8fe1f7e3ac88226b4f66d
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_314f7f82afeeb12c1301383bfe6c6bc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58efd46565f914100614802e53be6748
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_cf7575918270c02bf255e779335f8af9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd341c26c429be4deceb7b7802859f50
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3001679480075836, 0.03978395089507103, 0.13419044017791748, 0.09352608025074005], [0.39127877354621887, 0.4333009123802185, 0.06490619480609894, 0.38775861263275146], [0.39391446113586426, 0.4543362259864807, 0.011423652060329914, 0.07635422796010971], [0.39127877354621887, 0.4333009123802185, 0.06490619480609894, 0.38775861263275146], [0.39391446113586426, 0.4543362259864807, 0.011423652060329914, 0.07635422796010971], [0.4616011083126068, 0.20268231630325317, 0.38833072781562805, 0.20092904567718506], [0.4616011083126068, 0.20268231630325317, 0.38833072781562805, 0.20092904567718506]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.295341819524765, 0.11133721470832825, 0.12058486044406891, 0.14470656216144562], [0.2465321570634842, 0.16365154087543488, 0.0819966197013855, 0.021441368386149406], [0.0940665453672409, 0.44933849573135376, 0.40764477849006653, 0.22932206094264984], [0.2465321570634842, 0.16365154087543488, 0.0819966197013855, 0.021441368386149406], [0.0940665453672409, 0.44933849573135376, 0.40764477849006653, 0.22932206094264984], [0.23235131800174713, 0.08092588931322098, 0.4600076675415039, 0.27263739705085754], [0.23235131800174713, 0.08092588931322098, 0.4600076675415039, 0.27263739705085754]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_7297aa0a9ba9af38d40182533e39e015(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.33742555975914, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.09696357697248459], dtype='float32').reshape([6]),
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2f621e86face0be7bffc7001bf66377b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.2536962032318115, 0.2332039326429367, 0.22083275020122528, 0.2845509946346283, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_439698035d1b4ba85145e447e98d49e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.07290997356176376, 0.22891433537006378, 0.06531837582588196, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_7b7de9863b2f6e8b6c108f8c200d02da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2081550508737564, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.285847932100296], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.10902168601751328, 0.04308721795678139, 0.134864941239357, 0.4397190809249878, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_cd4fce2c5b26e9e3bd00d8172fb82783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.07290997356176376, 0.22891433537006378, 0.223669171333313, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0fdab695546d297d71114685b42aa2b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.4702686071395874, 0.36698785424232483, 0.134864941239357, 0.4465831220149994, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4f3ab3e7573e4dd7594ab4c5f2d54e2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0057981635c1b4b01c9f29c769235b00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_af37cf6c7d8d4e316b79815b756feba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.020324425771832466, 0.013810954988002777, -0.004959273152053356, 0.040052562952041626, -0.04735404625535011, -0.022230884060263634], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_71b093977455f98cddd6cce1ba5e9188(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.35703232884407043, 0.2805730104446411, 0.1354352831840515, 0.15389223396778107, 0.2500401735305786, 0.16624096035957336], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2881614863872528, 0.071312814950943, 0.06014230102300644, 0.2271801382303238, 0.12440825253725052, 0.28540241718292236], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8d77d12a9be5946c0945c2eda93c4187(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.17309994995594025, 0.36198240518569946, 0.30009588599205017, 0.1426411271095276, 0.36556705832481384, 0.08817589282989502], dtype='float32').reshape([6]),
                paddle.to_tensor([0.26979461312294006, 0.2617741525173187, 0.14019326865673065, 0.12376508861780167, 0.23937153816223145, 0.2517835199832916], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2dd435a3304c0a5902e41ddb561bc306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.040052223950624466, 0.026079408824443817, 0.06531837582588196, 0.23551833629608154], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_336cf92b62b370307b138f627949367d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2081550508737564, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.285847932100296], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1651548445224762, 0.10902168601751328, 0.04308721795678139, 0.0644494965672493, 0.4397190809249878, 0.07492171972990036], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8dea4d982184bbda99602ac8b0fc1b2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.9287320971488953, 0.14690853655338287, -0.13073183596134186, 0.15498313307762146, -0.28680527210235596, -1.3649654388427734], dtype='float32').reshape([6]),
                paddle.to_tensor([-1.1857959032058716, -1.0462806224822998, -0.9591996669769287, 1.0217697620391846, -0.31468695402145386, -1.3817603588104248], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_78bc811c809ba7d742f54125bb37b8fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1759, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1f215d139708233968525220359c89f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78bc811c809ba7d742f54125bb37b8fc
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e23b36e7efe2f91d47e719ee296f3fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c34cb26f95b9c844e8b6fc9d077a6a2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_1f215d139708233968525220359c89f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78bc811c809ba7d742f54125bb37b8fc
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_9258511bac62c77d151b86822e643225(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c9932fc45296674f2f74b8b06a5dade
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.3306543529033661, 0.3834773600101471, 0.49611005187034607, 0.11812765151262283, 0.478644460439682, 0.32757896184921265, 0.3514584004878998, 0.3204523026943207, 0.48325031995773315, 0.25242266058921814, 0.4594441056251526, 0.10603760182857513, 0.09884601831436157, 0.039876293390989304, 0.12443646788597107, 0.20525671541690826, 0.031096970662474632, 0.08158884942531586, 0.2653093934059143, 0.3785844147205353, 0.4728078544139862, 0.47888636589050293, 0.37857428193092346, 0.030704200267791748], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_2eb55c91a836056982e5b49298544827(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c9932fc45296674f2f74b8b06a5dade
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3306543529033661, 0.3834773600101471, 0.49611005187034607, 0.11812765151262283, 0.478644460439682, 0.32757896184921265, 0.3514584004878998, 0.3204523026943207, 0.48325031995773315, 0.25242266058921814, 0.4594441056251526, 0.10603760182857513, 0.09884601831436157, 0.039876293390989304, 0.12443646788597107, 0.20525671541690826, 0.031096970662474632, 0.08158884942531586, 0.2653093934059143, 0.3785844147205353, 0.4728078544139862, 0.47888636589050293, 0.37857428193092346, 0.030704200267791748], dtype='float32').reshape([24]),
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


    
    class PrimitiveOp_560628d1a6cbc05a7df5c44ad07d6ab3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1538, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ba77d450661cfda2e6af2683b1a4b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560628d1a6cbc05a7df5c44ad07d6ab3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b639c68824feae674a9217cfc6ba28e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef492e411a71dd985a40d5ca7e4f1834
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0ba77d450661cfda2e6af2683b1a4b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560628d1a6cbc05a7df5c44ad07d6ab3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a03afd939218f05bb4d6afdc889ea623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb4a5f8d2f9250767a759fceb26fb2b8
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.004606406204402447, 0.23123939335346222, 0.4518365263938904, 0.139329195022583], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_7ca94bb82f83e2614fea5459be6d68d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb4a5f8d2f9250767a759fceb26fb2b8
        def get_inputs(self):
            return [
                paddle.to_tensor([0.004606406204402447, 0.23123939335346222, 0.4518365263938904, 0.139329195022583], dtype='float32').reshape([4]),
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


    class TestPrimitiveOp_25cd2b14cfffd6aafcb1f1ff2840d22d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e17b6d4406f6436b786e9839fcef53
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3088763952255249, 0.12988963723182678, 0.4854060709476471, 0.29282501339912415], [0.15517044067382812, 0.20120646059513092, 0.17029418051242828, 0.21977749466896057], [0.40536436438560486, 0.4242015480995178, 0.1562468409538269, 0.3942132294178009], [0.4052164852619171, 0.16592980921268463, 0.16075582802295685, 0.11449424177408218], [0.4052164852619171, 0.16592980921268463, 0.16075582802295685, 0.11449424177408218], [0.40536436438560486, 0.4242015480995178, 0.1562468409538269, 0.3942132294178009]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.3819200098514557, 0.23831237852573395, 0.25062572956085205, 0.26940813660621643], [0.11866627633571625, 0.4657531976699829, 0.4957130253314972, 0.3083881437778473], [0.1402580738067627, 0.3681163191795349, 0.37202703952789307, 0.2937219440937042], [0.327421098947525, 0.23211318254470825, 0.1837078183889389, 0.3576776087284088], [0.327421098947525, 0.23211318254470825, 0.1837078183889389, 0.3576776087284088], [0.1402580738067627, 0.3681163191795349, 0.37202703952789307, 0.2937219440937042]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_be06c5799f61ca0f157a9a07a73d1a30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4678221344947815, 0.04469945654273033, 0.23884932696819305, 0.25630196928977966], [0.06454280763864517, 0.35123002529144287, 0.2817023694515228, 0.07664187997579575], [0.387455552816391, 0.08136474341154099, 0.36431193351745605, 0.0010941538494080305], [0.13462622463703156, 0.4734646677970886, 0.328313946723938, 0.43288931250572205], [0.4678221344947815, 0.04469945654273033, 0.23884932696819305, 0.25630196928977966]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.4540683627128601, 0.33153462409973145, 0.20022638142108917, 0.06467278301715851], [0.44031772017478943, 0.4055336117744446, 0.034483954310417175, 0.1595418006181717], [0.2573700547218323, 0.22866977751255035, 0.2128351330757141, 0.3317737579345703], [0.386608362197876, 0.10989176481962204, 0.45954629778862, 0.22013701498508453], [0.4540683627128601, 0.33153462409973145, 0.20022638142108917, 0.06467278301715851]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_fa78eeb41e7c029c5a4c471c4eb8b890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c47e42830e0306ce78a88fbacc03320c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_e3bc088c15e51acf75946b131c3acc8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2d02900099611dc3228b3cfe75b1c581(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ace172f4d1dc9b8bd0f65b3207e11fff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c47e42830e0306ce78a88fbacc03320c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1e8a0388eb17a8edfa851a1c0fd45e34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0011826035333797336]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_4eb56beda87f9dd90f49ca762c3440cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2d02900099611dc3228b3cfe75b1c581(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c7799b84f11fb29a5a8932a7ffddd1eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03425128385424614]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0011826036497950554]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_5f66a6210f942eaffc3565a2ddc08e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.9654726982116699]], dtype='float32').reshape([1, 1]),
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


    class TestPrimitiveOp_0c8c3a094f8dd10fdb004a1a5d651f9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.14439257979393005], [0.09650922566652298], [0.16213607788085938], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3966229259967804], [0.4306377172470093], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e0ee1ea5592dac54f3c18c3dcbf08a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e8a241d0052f2cb6b15c3523f29c1f06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.3532122075557709], [0.09650922566652298], [0.2863064706325531], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3966229259967804], [0.2040838748216629], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_a6289f03797d6e7462ce7ee9d9b0cd98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_21b2e32c70ee445c92110a8d8887418b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4361984133720398], [0.14439257979393005], [0.49514085054397583], [0.16213607788085938], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.4306377172470093], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_6ba4c7ed4fa5570d0b9625e516191278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_3ac4c6aeb49fff8dda46edcd7f6e5dbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24575075507164001], [-0.04205697402358055], [0.0034978860057890415], [0.003514286130666733], [0.16795028746128082], [0.1216726154088974]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.014378788881003857], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_857c1615577f831082a757da3b7e503c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4361984133720398], [0.3532122075557709], [0.49514085054397583], [0.2863064706325531], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.2040838748216629], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_611a58f55898dee2131d9cf4c10a293f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e8aa7d0befedd19e7952bd0d0cc6339c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09241148084402084], [0.023195823654532433], [-0.002818223787471652], [0.026250513270497322], [0.1769580990076065], [0.013878539204597473]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.24575075507164001], [-0.04205697402358055], [0.0034978860057890415], [0.003514286130666733], [0.1535715013742447], [0.1216726154088974]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_46c96b3e531ecccbff1fa0327b4654f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [0.0], [0.0], [0.09362927824258804], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-1.6593097448349], [2.813126802444458], [2.241166830062866], [0.8661250472068787], [0.13215894997119904], [-7.766961097717285]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_88a9d587998577509306f3725f1cf7bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7d91148da6f4d25d1ed84b8a21b1473
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20391754806041718, 0.31501132249832153, 0.07210913300514221, 0.2850325405597687], [0.42501765489578247, 0.20123079419136047, 0.21216268837451935, 0.4939834475517273], [0.10778137296438217, 0.2493751496076584, 0.11860713362693787, 0.3470500409603119], [0.18004286289215088, 0.1343468427658081, 0.10950658470392227, 0.3252163529396057]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.10165496915578842, 0.17966170608997345, 0.15234315395355225, 0.2450416386127472], [0.12599779665470123, 0.2939036786556244, 0.42129790782928467, 0.0379941388964653], [0.30939817428588867, 0.3820403516292572, 0.22087596356868744, 0.14539435505867004], [0.17156982421875, 0.10984878242015839, 0.1759537160396576, 0.06420376151800156]], dtype='float32').reshape([4, 4]),
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


    
    class PrimitiveOp_e737f7dbf79020313d984927edaaa1c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2135, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6217089a54f80f9215f2876fa544e19b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e737f7dbf79020313d984927edaaa1c7
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c91d096b4507a5aa804945af50b3506(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd12031a799b636ca53fb548ffa7c8df
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_6217089a54f80f9215f2876fa544e19b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e737f7dbf79020313d984927edaaa1c7
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6750ed160ab5f5ffce4f6f08dcc98cda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd341c26c429be4deceb7b7802859f50
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17218145728111267, 0.06576531380414963, 0.08751874417066574, 0.30011749267578125], [0.17218145728111267, 0.06576531380414963, 0.08751874417066574, 0.30011749267578125], [0.20134691894054413, 0.11920982599258423, 0.2738986015319824, 0.25143593549728394], [0.17379897832870483, 0.22546128928661346, 0.18680396676063538, 0.008240723982453346], [0.09319761395454407, 0.1083202138543129, 0.17002469301223755, 0.3804638981819153], [0.1670711785554886, 0.051270559430122375, 0.39242714643478394, 0.3678208291530609], [0.07703103125095367, 0.42867806553840637, 0.35202184319496155, 0.48333871364593506]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.014671501703560352, 0.33271241188049316, 0.4347265660762787, 0.0826963558793068], [0.014671501703560352, 0.33271241188049316, 0.4347265660762787, 0.0826963558793068], [0.050411611795425415, 0.2639580965042114, 0.4960781931877136, 0.15688583254814148], [0.04975507780909538, 0.2948163151741028, 0.31037741899490356, 0.3716491460800171], [0.07919828593730927, 0.49961045384407043, 0.35128462314605713, 0.17356255650520325], [0.1732870638370514, 0.1521768718957901, 0.08007995039224625, 0.22500772774219513], [0.4506492614746094, 0.35124221444129944, 0.08421099931001663, 0.22466899454593658]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_bf9c14e1ab7a79e2ae4816a9e9959569(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[4590, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2e4a3cdfa0b7db0b558a1df406aafdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf9c14e1ab7a79e2ae4816a9e9959569
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e956a8e6647d322d2ce45f032578f946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bbf8e1cff13f3d084ec01dea4c5076
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2e4a3cdfa0b7db0b558a1df406aafdf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf9c14e1ab7a79e2ae4816a9e9959569
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4ae10a6d9621cdf641676833691d665a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1042, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbbb9c4fdfd58ecedea32a095fb604f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae10a6d9621cdf641676833691d665a
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_259033fbe542dfd1b40aab42193a020a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77757fddfb5f4a482de4387208a1e8ba
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_dbbb9c4fdfd58ecedea32a095fb604f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae10a6d9621cdf641676833691d665a
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8e8efcaa720c7dabb100255c4a65bc73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e17b6d4406f6436b786e9839fcef53
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20043645799160004, 0.4294639527797699, 0.2814168930053711, 0.1474073827266693], [0.046092040836811066, 0.22925116121768951, 0.28526708483695984, 0.15180309116840363], [0.046092040836811066, 0.22925116121768951, 0.28526708483695984, 0.15180309116840363], [0.4408273696899414, 0.30982300639152527, 0.4029184579849243, 0.18092027306556702], [0.2481885403394699, 0.33687835931777954, 0.37247517704963684, 0.227286696434021], [0.4674718976020813, 0.22988873720169067, 0.3623161315917969, 0.3304578959941864]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.07959121465682983, 0.2769489586353302, 0.46637436747550964, 0.364219069480896], [0.03007488325238228, 0.23571008443832397, 0.4971219003200531, 0.3081081807613373], [0.03007488325238228, 0.23571008443832397, 0.4971219003200531, 0.3081081807613373], [0.08231888711452484, 0.19332893192768097, 0.23069065809249878, 0.4883212447166443], [0.22639085352420807, 0.29252418875694275, 0.15105053782463074, 0.4269372522830963], [0.44897162914276123, 0.1686171293258667, 0.49535754323005676, 0.23849567770957947]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_6dd1b6b298aedb31e5e0a10e389098c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5195df1055a103146ed731f087a9e76
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.06011958047747612, 27.418258666992188, 2.888007402420044, 1.416237473487854], [0.17004764080047607, 1.3946164846420288, 0.7928432822227478, 0.07617802172899246]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_f6d46dfd22de30f4d96d173342af6b05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de0324bed078bcbd34522cd4edfe7a49
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1.1455035209655762, 0.9406334161758423, 0.9402686357498169, 2.975639820098877], [9.551335334777832, 1.0109132528305054, 2.315927267074585, 0.2375415414571762]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_20d9f6cac4261186a723d2214e763807(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.3282714784145355], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_0ed7e64b3c57e63cf9fd54a5afa68003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.31906890869140625], [0.04625708982348442], [0.439250111579895], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_2f201dcb0bab420a97aa3d2c15140248(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.36288416385650635], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_a4ea1f67bf75e3ae07919b88a224fd31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.31906890869140625], [0.02482573315501213], [0.439250111579895], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f3b23a74af876fe5a6b96feb8dd3c91a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.3282714784145355], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b0bee815dac644579ecb0c923cb4666a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.04625708982348442], [0.1838442087173462], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c5e8d1bc61085a8c389afc874811151d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.005169573239982128], [0.06563594937324524], [0.019044138491153717], [0.1293209046125412], [-0.01654825359582901]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_83e312cccd8e0eea32d2afab9227e3ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.36288416385650635], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c4b2f904f4b069d94d462f52a301e75e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.02482573315501213], [0.1838442087173462], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_400302fd0c2d6190735df8af0d13b47f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.012724664062261581], [0.08693233877420425], [0.11657001823186874], [0.09260812401771545], [0.010530282743275166]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.005169573239982128], [0.06563594937324524], [0.019044138491153717], [0.1293209046125412], [-0.01654825359582901]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9af6b691ff8432545da1f884583273ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[1.406264066696167], [0.24497660994529724], [0.8366292119026184], [-0.3964315354824066], [2.5714917182922363]], dtype='float32').reshape([5, 1]),
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


    
    class PrimitiveOp_a098c0aed8cdc2509106138755e74a05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2339, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18b5a94424800120ce0b772345ebe28d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a098c0aed8cdc2509106138755e74a05
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae7acdab9dee85c00c6758f971918b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02f7c478bd6a616eccc7742a17e926ce
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_18b5a94424800120ce0b772345ebe28d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a098c0aed8cdc2509106138755e74a05
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f3e7b2128dd41939392778d49606331(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[3063, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_314476ef9fde3896d7405e88b5189b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f3e7b2128dd41939392778d49606331
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2a563a8468194b0c46af06c275c59a48(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a90d3a2d55245d256ab5e244e6207cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a563a8468194b0c46af06c275c59a48
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_314476ef9fde3896d7405e88b5189b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f3e7b2128dd41939392778d49606331
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_482a341d1940f5c07fccc4098f2413dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[3822, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b571e11f01545f700dbbeb0cba8e9d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_482a341d1940f5c07fccc4098f2413dd
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_53d23d9805f042835e1432063dabc3d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b3c111d89ae297bb72cfc7aeb33c85d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53d23d9805f042835e1432063dabc3d6
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b571e11f01545f700dbbeb0cba8e9d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_482a341d1940f5c07fccc4098f2413dd
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a2b33ea7dfe9a57937dd5a4855f7b661(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ee6c518cc2c17722e81ecf9fb65043
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.34141385555267334, 0.08070738613605499, 0.03860168904066086, 0.49608147144317627, 0.41707998514175415, 0.07405710965394974, 0.39049777388572693, 0.14312376081943512, 0.4575801193714142, 0.44072431325912476, 0.48256993293762207, 0.10408809781074524, 0.3957173824310303, 0.045056845992803574, 0.2210846096277237, 0.11286043375730515, 0.109720878303051, 0.17332060635089874, 0.47122129797935486, 0.48317795991897583], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_5b07ca890c09d5abbd23f6cb16bd0c07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ee6c518cc2c17722e81ecf9fb65043
        def get_inputs(self):
            return [
                paddle.to_tensor([0.34141385555267334, 0.08070738613605499, 0.03860168904066086, 0.49608147144317627, 0.41707998514175415, 0.07405710965394974, 0.39049777388572693, 0.14312376081943512, 0.4575801193714142, 0.44072431325912476, 0.48256993293762207, 0.10408809781074524, 0.3957173824310303, 0.045056845992803574, 0.2210846096277237, 0.11286043375730515, 0.109720878303051, 0.17332060635089874, 0.47122129797935486, 0.48317795991897583], dtype='float32').reshape([20]),
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


    class TestPrimitiveOp_035c4d267148a0d7c49ca5dba2d6ab5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14522317051887512], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_dfc3e0027752797d8f24ff9ded1cd80c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.440621554851532], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.32093942165374756], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_094f28e057f176b468d3b77ff79d66f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7cea4b0ef5b1e3a1d25e8fbf764a0723(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.48723578453063965], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.32093942165374756], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_16aa1a18ae04ebbd05bc1a7f2b1907f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14522317051887512], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f399482f098dc1086ef59bcca0b55956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28212907910346985], [0.440621554851532], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_ae2bcfab590cc8443fdbec55aa5963a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.04221170023083687], [0.1649954915046692], [-0.0011601648293435574], [-0.019290968775749207]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.02299167960882187], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_0135ce42d9778bd6ba7a0d2929cc4e7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a31742bb67104005dc95416e09442638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28212907910346985], [0.48723578453063965], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_e9d8f4490d40fec77862362d647e7da3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.011106634512543678], [0.16120393574237823], [0.0076672681607306], [0.06769919395446777]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.04221170023083687], [0.14200380444526672], [-0.0011601647129282355], [-0.019290968775749207]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_b692f2eac60ddba4ece828dc05d40527(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.16190889477729797], [-0.0], [-0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[4.800584316253662], [0.11910460889339447], [1.1513140201568604], [1.2849512100219727]], dtype='float32').reshape([4, 1]),
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


    
    class PrimitiveOp_42d97862ec88c7d6ae17f4b985bc6acc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2057, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c83dfa96ce59ac716489c196dfca1b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42d97862ec88c7d6ae17f4b985bc6acc
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c5e58587b1ca796433546b1ec5268395(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b2a0ed160ddbfedb10379563bb74a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5e58587b1ca796433546b1ec5268395
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c83dfa96ce59ac716489c196dfca1b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42d97862ec88c7d6ae17f4b985bc6acc
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_db15486c2a434339f24fe2619263b9b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18006691336631775, 0.47249430418014526, 0.2704814374446869, 0.28356489539146423], [0.1444801688194275, 0.26884371042251587, 0.32883790135383606, 0.40756726264953613], [0.40327703952789307, 0.16330288350582123, 0.4570107161998749, 0.3071627914905548], [0.40327703952789307, 0.16330288350582123, 0.4570107161998749, 0.3071627914905548], [0.4027557671070099, 0.4091954827308655, 0.3253519833087921, 0.04934440553188324]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.18216904997825623, 0.3664987087249756, 0.09633531421422958, 0.16741898655891418], [0.432188481092453, 0.08552742004394531, 0.4801865220069885, 0.10212612897157669], [0.09275709837675095, 0.3269622027873993, 0.2648838758468628, 0.37689346075057983], [0.09275709837675095, 0.3269622027873993, 0.2648838758468628, 0.37689346075057983], [0.1461302638053894, 0.10133280605077744, 0.3515809178352356, 0.4469776153564453]], dtype='float32').reshape([5, 4]),
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


    
    class PrimitiveOp_c4ff6e8bb8cb2a43d9b86e8e3deedbce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[4189, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb5cd928bd2f2efdd4a24cea2f29556d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4ff6e8bb8cb2a43d9b86e8e3deedbce
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536812ec33e7ff4a53ef94f5d345aadf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a57fca7f1089519aa018a3a7e06960a9
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_cb5cd928bd2f2efdd4a24cea2f29556d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4ff6e8bb8cb2a43d9b86e8e3deedbce
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c394d762ff521f652d29f0070d9984da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd341c26c429be4deceb7b7802859f50
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07471240311861038, 0.1784989833831787, 0.0961991548538208, 0.2725340723991394], [0.4185662865638733, 0.41786858439445496, 0.3821410536766052, 0.3345850706100464], [0.39434146881103516, 0.07983526587486267, 0.014400581829249859, 0.29254430532455444], [0.07471240311861038, 0.1784989833831787, 0.0961991548538208, 0.2725340723991394], [0.348670095205307, 0.276864618062973, 0.2532390356063843, 0.433200865983963], [0.07939526438713074, 0.25467807054519653, 0.49734240770339966, 0.2597086429595947], [0.348670095205307, 0.276864618062973, 0.2532390356063843, 0.433200865983963]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.1009109765291214, 0.05554174259305, 0.07083551585674286, 0.32371416687965393], [0.15809713304042816, 0.25184744596481323, 0.13013799488544464, 0.4204864799976349], [0.12200950086116791, 0.45094767212867737, 0.051762133836746216, 0.018873345106840134], [0.1009109765291214, 0.05554174259305, 0.07083551585674286, 0.32371416687965393], [0.3411698639392853, 0.017546646296977997, 0.26674339175224304, 0.05254773423075676], [0.46994319558143616, 0.45695093274116516, 0.27365565299987793, 0.011935419403016567], [0.3411698639392853, 0.017546646296977997, 0.26674339175224304, 0.05254773423075676]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_d0d91251eac078da8f7df233424fcadc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4618493914604187]], [[0.4058695137500763]], [[0.08289115130901337]], [[0.26278284192085266]], [[0.23262260854244232]], [[0.013348624110221863]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7939244508743286]], [[0.8136937618255615]], [[0.6444892287254333]], [[0.7144414186477661]], [[0.738219141960144]], [[0.6364917755126953]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_1cd96bd0b741453e69e3abc09e97b2db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.23339112102985382]], [[0.22922959923744202]], [[0.20036675035953522]], [[0.05275467038154602]], [[0.09179128706455231]], [[0.08593977987766266]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6195000410079956]], [[0.655858039855957]], [[0.6888777017593384]], [[0.7641112804412842]], [[0.5334721207618713]], [[0.6528226137161255]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_806d72dbfcc3f93f944a2306ff791e7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.29278796911239624, 0.12469929456710815]], [[0.19302858412265778, 0.39783740043640137]], [[0.1291770488023758, 0.45455050468444824]], [[0.012822974473237991, 0.04444433003664017]], [[0.32388806343078613, 0.1391058713197708]], [[0.46277740597724915, 0.001989981159567833]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.42187127470970154, 0.18227285146713257]], [[0.25669822096824646, 0.37202227115631104]], [[0.17325818538665771, 0.39465776085853577]], [[0.18558315932750702, 0.20827585458755493]], [[0.3987077474594116, 0.1856575310230255]], [[0.08304066210985184, 0.22850774228572845]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_8dcdfb700c44c0048d31790fc509e840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.46550989151000977, 0.1697573959827423]], [[0.13966119289398193, 0.4147696793079376]], [[0.2085360884666443, 0.08584899455308914]], [[0.2911050617694855, 0.42420315742492676]], [[0.1459295153617859, 0.45230230689048767]], [[0.01436071377247572, 0.10688220709562302]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.42187127470970154, 0.18227285146713257]], [[0.25669822096824646, 0.37202227115631104]], [[0.17325818538665771, 0.39465776085853577]], [[0.18558315932750702, 0.20827585458755493]], [[0.3987077474594116, 0.1856575310230255]], [[0.08304066210985184, 0.22850774228572845]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_7ebcec0052ef2cd69031d7e6a94c5c5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.4262748956680298, 0.1736283153295517]], [[0.1497272402048111, 0.43034040927886963]], [[0.4598858952522278, 0.04210919141769409]], [[0.2074327915906906, 0.2813599109649658]], [[0.46443691849708557, 0.28645503520965576]], [[0.16248425841331482, 0.24488294124603271]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_ce4b3f83834348c3a5df8c24a6b3fb88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.4765012264251709, 0.2170112580060959, 0.4702689051628113, 0.4369574785232544, 0.244343563914299, 0.49530693888664246, 0.3121092915534973, 0.0963035523891449, 0.38352274894714355, 0.009155333042144775, 0.4616371691226959, 0.020641636103391647, 0.15587159991264343, 0.02491425909101963, 0.443154513835907, 0.24451406300067902], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_9cb18e4462f220e11b7fa62202ec5a26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4765012264251709, 0.2170112580060959, 0.4702689051628113, 0.4369574785232544, 0.244343563914299, 0.49530693888664246, 0.3121092915534973, 0.0963035523891449, 0.38352274894714355, 0.009155333042144775, 0.4616371691226959, 0.020641636103391647, 0.15587159991264343, 0.02491425909101963, 0.443154513835907, 0.24451406300067902], dtype='float32').reshape([16]),
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


    class TestPrimitiveOp_8116507d5f932d238992ee3be63dfdb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e01f581e0f1da81e0d1d38c1d920b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8116507d5f932d238992ee3be63dfdb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d46df5ec11068e5d2ebe6414441ec4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.45848238468170166, 0.34971383213996887, 0.3945930004119873, 0.447701632976532], [0.41079744696617126, 0.0009267344721592963, 0.08779380470514297, 0.06212601438164711], [0.2339312881231308, 0.2007937729358673, 0.15278489887714386, 0.3617117404937744], [0.4200783371925354, 0.41086649894714355, 0.39261895418167114, 0.24603603780269623], [0.32407864928245544, 0.22759559750556946, 0.45276063680648804, 0.4318024814128876]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.4423430263996124, 0.3251437842845917, 0.2146957665681839, 0.06303687393665314], [0.09525219351053238, 0.11553490161895752, 0.0325784906744957, 0.13175402581691742], [0.2793131172657013, 0.4104779064655304, 0.26631882786750793, 0.19468954205513], [0.10130643099546432, 0.19705356657505035, 0.33797094225883484, 0.3692222237586975], [0.18079429864883423, 0.10179086029529572, 0.4477527141571045, 0.4130363166332245]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_18d12dfe556373bfed233390542ddf09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015340379439294338, 0.024809498339891434, 0.4150008261203766, 0.38157200813293457], [0.4599052965641022, 0.14341849088668823, 0.12601007521152496, 0.34754735231399536], [0.49729982018470764, 0.24716007709503174, 0.34805798530578613, 0.10370808094739914], [0.4599052965641022, 0.14341849088668823, 0.12601007521152496, 0.34754735231399536], [0.49729982018470764, 0.24716007709503174, 0.34805798530578613, 0.10370808094739914]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.18711674213409424, 0.19057035446166992, 0.2047814428806305, 0.4759475886821747], [0.09319417178630829, 0.05631349980831146, 0.031228844076395035, 0.19341900944709778], [0.08189824223518372, 0.08736616373062134, 0.3658214509487152, 0.29241862893104553], [0.09319417178630829, 0.05631349980831146, 0.031228844076395035, 0.19341900944709778], [0.08189824223518372, 0.08736616373062134, 0.3658214509487152, 0.29241862893104553]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_712e1be40545814c3fca2aa24262ae9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.2974533438682556], [0.04537849500775337], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_b7038710d38ad77bf74f4448a686ad47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09990614652633667], [0.2677699625492096], [0.015402782708406448], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.1261843889951706], [0.030494073405861855], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.04978129267692566], [0.2015986442565918], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.3369775414466858], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_4bbde2670142f95da11786eb93148a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.2974533438682556], [0.04537849500775337], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.40820974111557007], [0.3767457604408264], [0.20149092376232147], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.24313530325889587]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e99402103a964419db7cc66e3e3bf0e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.2677699625492096], [0.015402782708406448], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.1261843889951706], [0.05905040726065636], [0.30280032753944397]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.04978129267692566], [0.042689915746450424], [0.1670447289943695], [0.3255453109741211], [0.22168521583080292], [0.11604061722755432], [0.43528881669044495]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_a0e6a022bb7c5df5826a182ea11c228d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.39238229393959045], [0.158734992146492], [0.4127102792263031], [0.41945791244506836], [0.29879409074783325], [0.013535123318433762], [0.1018265038728714], [0.21279259026050568], [0.030626868829131126]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4038843512535095], [0.44629916548728943], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.179908886551857], [0.3791463375091553], [0.4934486150741577], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_be878a040d4295576e8eae326a802ac8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09990614652633667], [0.32696646451950073], [0.2620046138763428], [0.3387252688407898], [0.2716210186481476], [0.1099492609500885], [0.29109078645706177], [0.030494073405861855], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.22095853090286255], [0.2874982953071594], [0.0034736385568976402], [0.2015986442565918], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.3369775414466858], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2676e1f6d07268be9c7acd4fdd4c7967(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05231598764657974], [0.009299254976212978], [0.10837691277265549], [-0.07017733156681061], [0.08777499198913574], [-0.016110287979245186], [-0.0599064826965332], [0.08122386783361435], [-0.026601403951644897]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.010175604373216629], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_da08024382ac218db477bc6350f59684(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41298598051071167], [0.48954424262046814], [0.4127102792263031], [0.41945791244506836], [0.4212323725223541], [0.14318227767944336], [0.2683485746383667], [0.40157246589660645], [0.34450793266296387]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2521946430206299], [0.19013427197933197], [0.008235457353293896], [0.14164923131465912], [0.18166889250278473], [0.07500499486923218], [0.328750878572464], [0.31747785210609436], [0.07347828894853592]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_d571809b30dca09539b5ba28c8160331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36636999249458313], [0.32696646451950073], [0.2620046138763428], [0.36943402886390686], [0.43435823917388916], [0.2168995440006256], [0.29109078645706177], [0.05905040726065636], [0.40721428394317627]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.04966363683342934], [0.19880470633506775], [0.0034736385568976402], [0.042689915746450424], [0.023722784593701363], [0.057638633996248245], [0.05427054315805435], [0.11604061722755432], [0.09985696524381638]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_715886cc893a6fd3b4f71859b5031113(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05092363804578781], [0.038372911512851715], [0.10456927120685577], [0.090772345662117], [0.09837325662374496], [0.010857976041734219], [-0.014304488897323608], [-0.004792569670826197], [0.08330294489860535]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.05231598764657974], [0.009299254976212978], [0.10837691277265549], [-0.07017733156681061], [0.07759939134120941], [-0.016110287979245186], [-0.0599064826965332], [0.08122386783361435], [-0.026601403951644897]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e63d7544515b5854633cde355353a35f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [-0.0], [0.13112995028495789], [-0.0], [-0.0], [0.0], [-0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.02734191156923771], [0.7576609253883362], [-0.03641262277960777], [1.7731136083602905], [0.2111739069223404], [2.4837284088134766], [-3.1879498958587646], [17.947874069213867], [1.3193333148956299]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_1af58d553c5e7b6ec645021f13cf8a4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5822a565ae043a4fa59022614565406(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.21194247901439667]], [[0.3961002826690674]], [[0.4367770254611969]], [[0.30254191160202026]], [[0.17525893449783325]], [[0.3245100975036621]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6667161583900452]], [[0.8103682398796082]], [[0.75272536277771]], [[0.6059465408325195]], [[0.7161102294921875]], [[0.8132511377334595]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_a0de73b450c16800ce9c578aeb02393c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.12078697234392166]], [[0.08913929015398026]], [[0.149296835064888]], [[0.39876019954681396]], [[0.13448122143745422]], [[0.1268261820077896]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.5343883037567139]], [[0.513643741607666]], [[0.5729232430458069]], [[0.5704134106636047]], [[0.7160937786102295]], [[0.7558856010437012]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_3938bc10f243ced1b56a73317bb0e0ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4439a4997989fbdc6ad1892ef459e502(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_586db0d5a0741a8921fbf02e71879e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4439a4997989fbdc6ad1892ef459e502(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e814782cf8084e98910d1afb21c39abf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3001679480075836, 0.03978395089507103, 0.13419044017791748, 0.09352608025074005], [0.39127877354621887, 0.4333009123802185, 0.06490619480609894, 0.38775861263275146], [0.39391446113586426, 0.4543362259864807, 0.011423652060329914, 0.07635422796010971], [0.39127877354621887, 0.4333009123802185, 0.06490619480609894, 0.38775861263275146], [0.39391446113586426, 0.4543362259864807, 0.011423652060329914, 0.07635422796010971], [0.4616011083126068, 0.20268231630325317, 0.38833072781562805, 0.20092904567718506], [0.4616011083126068, 0.20268231630325317, 0.38833072781562805, 0.20092904567718506]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.295341819524765, 0.11133721470832825, 0.12058486044406891, 0.14470656216144562], [0.2465321570634842, 0.16365154087543488, 0.0819966197013855, 0.021441368386149406], [0.0940665453672409, 0.44933849573135376, 0.40764477849006653, 0.22932206094264984], [0.2465321570634842, 0.16365154087543488, 0.0819966197013855, 0.021441368386149406], [0.0940665453672409, 0.44933849573135376, 0.40764477849006653, 0.22932206094264984], [0.23235131800174713, 0.08092588931322098, 0.4600076675415039, 0.27263739705085754], [0.23235131800174713, 0.08092588931322098, 0.4600076675415039, 0.27263739705085754]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_b11dbe1bc6ebf3a37e473d7536553d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.33742555975914, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.09696357697248459], dtype='float32').reshape([6]),
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c4c69ccf0263e70e002a9d67a4eed8c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.2536962032318115, 0.2332039326429367, 0.22083275020122528, 0.2845509946346283, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8fb6832185ca70dee28293d7ddf5f8cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.07290997356176376, 0.22891433537006378, 0.06531837582588196, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_9fab32bc0be449931b7f7343dbac446f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2081550508737564, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.285847932100296], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.10902168601751328, 0.04308721795678139, 0.134864941239357, 0.4397190809249878, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ad6864c8b87a8d9898121090f34a3d4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2057289332151413, 0.09391630440950394, 0.047374628484249115, 0.2254459410905838, 0.18349812924861908, 0.12224911153316498], dtype='float32').reshape([6]),
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.07290997356176376, 0.22891433537006378, 0.223669171333313, 0.44855570793151855], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e1d2fd63be576e1d50a05983861bd7fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.41452664136886597, 0.23729932308197021, 0.11266523599624634, 0.0390239953994751, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3314341604709625, 0.4702686071395874, 0.36698785424232483, 0.134864941239357, 0.4465831220149994, 0.21771910786628723], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8d8cb710c6b9af296a20d8014b8b5240(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.37663909792900085, 0.09341233968734741, 0.040052223950624466, 0.026079408824443817, 0.223669171333313, 0.23551833629608154], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8303440e5665b5050f220e36b8ca9cca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1810450553894043, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.10143007338047028], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1651548445224762, 0.4702686071395874, 0.36698785424232483, 0.0644494965672493, 0.4465831220149994, 0.07492171972990036], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0f6edb6a2b08fc91d6acc22d6de7a1d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.020324425771832466, 0.013810954988002777, -0.004959273152053356, 0.040052562952041626, -0.04735404625535011, -0.022230884060263634], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, -0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ad20ab878ccb3247d8f11388f309854c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.35703232884407043, 0.2805730104446411, 0.1354352831840515, 0.15389223396778107, 0.2500401735305786, 0.16624096035957336], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2881614863872528, 0.071312814950943, 0.06014230102300644, 0.2271801382303238, 0.12440825253725052, 0.28540241718292236], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e0f87e8fb8a595ae8b5c46914fcad387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.17309994995594025, 0.36198240518569946, 0.30009588599205017, 0.1426411271095276, 0.36556705832481384, 0.08817589282989502], dtype='float32').reshape([6]),
                paddle.to_tensor([0.26979461312294006, 0.2617741525173187, 0.14019326865673065, 0.12376508861780167, 0.23937153816223145, 0.2517835199832916], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_74ac441ef9838f0602f654f4679f615b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37663909792900085, 0.4677337110042572, 0.23081833124160767, 0.2817050516605377, 0.27641114592552185, 0.23551833629608154], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3705940544605255, 0.04870932921767235, 0.040052223950624466, 0.026079408824443817, 0.06531837582588196, 0.23551833629608154], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_017cd302cd3dbebf3341972ad2d2f4c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2081550508737564, 0.4702686071395874, 0.36698785424232483, 0.22083275020122528, 0.4465831220149994, 0.285847932100296], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1651548445224762, 0.10902168601751328, 0.04308721795678139, 0.0644494965672493, 0.4397190809249878, 0.07492171972990036], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ec003c842b6388bd0b10a3955bcb761c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.9287320971488953, 0.14690853655338287, -0.13073183596134186, 0.15498313307762146, -0.28680527210235596, -1.3649654388427734], dtype='float32').reshape([6]),
                paddle.to_tensor([-1.1857959032058716, -1.0462806224822998, -0.9591996669769287, 1.0217697620391846, -0.31468695402145386, -1.3817603588104248], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f38c7a8017e7f4463cfc1892356a7388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50049e34db7cb2344640872b446f6345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f38c7a8017e7f4463cfc1892356a7388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3738802e3b2affa6656be927be90b8b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8265d2b3756980f2d194236a95721671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.3306543529033661, 0.3834773600101471, 0.49611005187034607, 0.11812765151262283, 0.478644460439682, 0.32757896184921265, 0.3514584004878998, 0.3204523026943207, 0.48325031995773315, 0.25242266058921814, 0.4594441056251526, 0.10603760182857513, 0.09884601831436157, 0.039876293390989304, 0.12443646788597107, 0.20525671541690826, 0.031096970662474632, 0.08158884942531586, 0.2653093934059143, 0.3785844147205353, 0.4728078544139862, 0.47888636589050293, 0.37857428193092346, 0.030704200267791748], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_fa7d058ea1700290211a447a8442a35f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3306543529033661, 0.3834773600101471, 0.49611005187034607, 0.11812765151262283, 0.478644460439682, 0.32757896184921265, 0.3514584004878998, 0.3204523026943207, 0.48325031995773315, 0.25242266058921814, 0.4594441056251526, 0.10603760182857513, 0.09884601831436157, 0.039876293390989304, 0.12443646788597107, 0.20525671541690826, 0.031096970662474632, 0.08158884942531586, 0.2653093934059143, 0.3785844147205353, 0.4728078544139862, 0.47888636589050293, 0.37857428193092346, 0.030704200267791748], dtype='float32').reshape([24]),
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


    class TestPrimitiveOp_f5d3d8a1a6541234f99903ac7292db50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b38ad5b87fd4c0a1bee15743bac3f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f5d3d8a1a6541234f99903ac7292db50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_1c1f31afa1d7619193bdc80164594255(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.004606406204402447, 0.23123939335346222, 0.4518365263938904, 0.139329195022583], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_0a1c7c9df8dddeb4b34e5c991ea9f374(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.004606406204402447, 0.23123939335346222, 0.4518365263938904, 0.139329195022583], dtype='float32').reshape([4]),
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


    class TestPrimitiveOp_4c37a7b9139b876a4a2f0d831b63f650(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3088763952255249, 0.12988963723182678, 0.4854060709476471, 0.29282501339912415], [0.15517044067382812, 0.20120646059513092, 0.17029418051242828, 0.21977749466896057], [0.40536436438560486, 0.4242015480995178, 0.1562468409538269, 0.3942132294178009], [0.4052164852619171, 0.16592980921268463, 0.16075582802295685, 0.11449424177408218], [0.4052164852619171, 0.16592980921268463, 0.16075582802295685, 0.11449424177408218], [0.40536436438560486, 0.4242015480995178, 0.1562468409538269, 0.3942132294178009]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.3819200098514557, 0.23831237852573395, 0.25062572956085205, 0.26940813660621643], [0.11866627633571625, 0.4657531976699829, 0.4957130253314972, 0.3083881437778473], [0.1402580738067627, 0.3681163191795349, 0.37202703952789307, 0.2937219440937042], [0.327421098947525, 0.23211318254470825, 0.1837078183889389, 0.3576776087284088], [0.327421098947525, 0.23211318254470825, 0.1837078183889389, 0.3576776087284088], [0.1402580738067627, 0.3681163191795349, 0.37202703952789307, 0.2937219440937042]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_f8382f69cba3292720aa15e70db2a675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4678221344947815, 0.04469945654273033, 0.23884932696819305, 0.25630196928977966], [0.06454280763864517, 0.35123002529144287, 0.2817023694515228, 0.07664187997579575], [0.387455552816391, 0.08136474341154099, 0.36431193351745605, 0.0010941538494080305], [0.13462622463703156, 0.4734646677970886, 0.328313946723938, 0.43288931250572205], [0.4678221344947815, 0.04469945654273033, 0.23884932696819305, 0.25630196928977966]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.4540683627128601, 0.33153462409973145, 0.20022638142108917, 0.06467278301715851], [0.44031772017478943, 0.4055336117744446, 0.034483954310417175, 0.1595418006181717], [0.2573700547218323, 0.22866977751255035, 0.2128351330757141, 0.3317737579345703], [0.386608362197876, 0.10989176481962204, 0.45954629778862, 0.22013701498508453], [0.4540683627128601, 0.33153462409973145, 0.20022638142108917, 0.06467278301715851]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_73ece6d168a08bd3fa887a7c95aa72af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bee3f060aceabee63629beb3f0dccf5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9c32dfd5a108b6241abc2e187ce297d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_322ae80d88857285e2afe610cff5c342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.2060057669878006]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b0e6a49df85f5cc66f6606b92ee22a05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1846d81dda1f23fee4ee75e86e124ebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08478929847478867]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9c32dfd5a108b6241abc2e187ce297d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40578794479370117]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.3662005662918091]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_c54ce2c76fb2d05f1bb005484a3f464f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0011826035333797336]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_a7bf542e908842b46d8a5864c04fdd8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20535515248775482]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.05126293748617172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b0e6a49df85f5cc66f6606b92ee22a05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.491769939661026]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.26949211955070496]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_8e07ae180a1a2c837d0ef31bf19d42ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03425128385424614]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0011826036497950554]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_5236ea01db42b87f4d7210deb87c22c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.9654726982116699]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_841d1f35a585136b954a538ebd908f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.14439257979393005], [0.09650922566652298], [0.16213607788085938], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3966229259967804], [0.4306377172470093], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_92395d4ac97f39198c55d4cba1e9aeab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_b0738fdb03fefeae5e31e71b827d2ff0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05556122213602066], [0.3532122075557709], [0.09650922566652298], [0.2863064706325531], [0.4242008626461029], [0.07322079688310623]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3966229259967804], [0.2040838748216629], [0.2837878465652466], [0.1662617325782776], [0.08668506145477295], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ff82bf91aa7fc5faa224a0ff4fa1e111(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04935036972165108], [0.401021808385849], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.49894431233406067], [0.3844832479953766], [0.21334876120090485], [0.25372934341430664], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_cc6e32649afca92bfce4b82071455340(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4361984133720398], [0.14439257979393005], [0.49514085054397583], [0.16213607788085938], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.4306377172470093], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.47490552067756653]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d33ebc43fb70346fdeb41db34e1baac2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.14623917639255524], [0.0019387512002140284], [0.4178217053413391], [0.1282878816127777]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.3752198815345764], [0.4434046745300293]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_e2a800767c8c145c5bc88710bbdbf19e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.24575075507164001], [-0.04205697402358055], [0.0034978860057890415], [0.003514286130666733], [0.16795028746128082], [0.1216726154088974]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.014378788881003857], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_23948f1b1b43f4396d9c6278e5a7c8e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4361984133720398], [0.3532122075557709], [0.49514085054397583], [0.2863064706325531], [0.43578216433525085], [0.15172600746154785]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.17277514934539795], [0.2040838748216629], [0.21807396411895752], [0.11639563739299774], [0.04053834080696106], [0.3346942663192749]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_6db3dec6f7c0d352c90b9dc4779ca4a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4443901479244232], [0.49830520153045654], [0.15970346331596375], [0.36091819405555725], [0.45432034134864807], [0.24458400905132294]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.09358027577400208], [0.3427625000476837], [0.1698751002550125], [0.2064223736524582], [0.006601519882678986], [0.32043617963790894]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d2e0ed2cf6488071195df7349be43fa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09241148084402084], [0.023195823654532433], [-0.002818223787471652], [0.026250513270497322], [0.1769580990076065], [0.013878539204597473]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.24575075507164001], [-0.04205697402358055], [0.0034978860057890415], [0.003514286130666733], [0.1535715013742447], [0.1216726154088974]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_40bfc2224a371d51189d33d508137c5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [0.0], [0.0], [0.09362927824258804], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-1.6593097448349], [2.813126802444458], [2.241166830062866], [0.8661250472068787], [0.13215894997119904], [-7.766961097717285]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ddb143a473b351c891d3af24a72c8b43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20391754806041718, 0.31501132249832153, 0.07210913300514221, 0.2850325405597687], [0.42501765489578247, 0.20123079419136047, 0.21216268837451935, 0.4939834475517273], [0.10778137296438217, 0.2493751496076584, 0.11860713362693787, 0.3470500409603119], [0.18004286289215088, 0.1343468427658081, 0.10950658470392227, 0.3252163529396057]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.10165496915578842, 0.17966170608997345, 0.15234315395355225, 0.2450416386127472], [0.12599779665470123, 0.2939036786556244, 0.42129790782928467, 0.0379941388964653], [0.30939817428588867, 0.3820403516292572, 0.22087596356868744, 0.14539435505867004], [0.17156982421875, 0.10984878242015839, 0.1759537160396576, 0.06420376151800156]], dtype='float32').reshape([4, 4]),
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


    class TestPrimitiveOp_828d9e62262c453c0be465805a9984b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7c99b4b8b190cd1a7798ea18635aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_828d9e62262c453c0be465805a9984b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67328a2fbcb4caf2961f75be9a29808f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17218145728111267, 0.06576531380414963, 0.08751874417066574, 0.30011749267578125], [0.17218145728111267, 0.06576531380414963, 0.08751874417066574, 0.30011749267578125], [0.20134691894054413, 0.11920982599258423, 0.2738986015319824, 0.25143593549728394], [0.17379897832870483, 0.22546128928661346, 0.18680396676063538, 0.008240723982453346], [0.09319761395454407, 0.1083202138543129, 0.17002469301223755, 0.3804638981819153], [0.1670711785554886, 0.051270559430122375, 0.39242714643478394, 0.3678208291530609], [0.07703103125095367, 0.42867806553840637, 0.35202184319496155, 0.48333871364593506]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.014671501703560352, 0.33271241188049316, 0.4347265660762787, 0.0826963558793068], [0.014671501703560352, 0.33271241188049316, 0.4347265660762787, 0.0826963558793068], [0.050411611795425415, 0.2639580965042114, 0.4960781931877136, 0.15688583254814148], [0.04975507780909538, 0.2948163151741028, 0.31037741899490356, 0.3716491460800171], [0.07919828593730927, 0.49961045384407043, 0.35128462314605713, 0.17356255650520325], [0.1732870638370514, 0.1521768718957901, 0.08007995039224625, 0.22500772774219513], [0.4506492614746094, 0.35124221444129944, 0.08421099931001663, 0.22466899454593658]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_d01e9b09a4b38d6f31eb68c2069977d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78d952d750dd1afa22a0ea0706722c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d01e9b09a4b38d6f31eb68c2069977d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2581809a66322e2a2f5755812c86bb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c43ff6a97652fea07a0a94547d9346b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b2581809a66322e2a2f5755812c86bb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a89dfe01d48a19b39a14b931a6f22cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_968be798066db33bf4ba449d629257c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20043645799160004, 0.4294639527797699, 0.2814168930053711, 0.1474073827266693], [0.046092040836811066, 0.22925116121768951, 0.28526708483695984, 0.15180309116840363], [0.046092040836811066, 0.22925116121768951, 0.28526708483695984, 0.15180309116840363], [0.4408273696899414, 0.30982300639152527, 0.4029184579849243, 0.18092027306556702], [0.2481885403394699, 0.33687835931777954, 0.37247517704963684, 0.227286696434021], [0.4674718976020813, 0.22988873720169067, 0.3623161315917969, 0.3304578959941864]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.07959121465682983, 0.2769489586353302, 0.46637436747550964, 0.364219069480896], [0.03007488325238228, 0.23571008443832397, 0.4971219003200531, 0.3081081807613373], [0.03007488325238228, 0.23571008443832397, 0.4971219003200531, 0.3081081807613373], [0.08231888711452484, 0.19332893192768097, 0.23069065809249878, 0.4883212447166443], [0.22639085352420807, 0.29252418875694275, 0.15105053782463074, 0.4269372522830963], [0.44897162914276123, 0.1686171293258667, 0.49535754323005676, 0.23849567770957947]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_88e921bb69be7e4b9e3d5065b5f360d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.06011958047747612, 27.418258666992188, 2.888007402420044, 1.416237473487854], [0.17004764080047607, 1.3946164846420288, 0.7928432822227478, 0.07617802172899246]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_9f7d0d426fc845676b9f730410fe5f34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1.1455035209655762, 0.9406334161758423, 0.9402686357498169, 2.975639820098877], [9.551335334777832, 1.0109132528305054, 2.315927267074585, 0.2375415414571762]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_f27f94c3ae1973a2e8432dd80a318111(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.3282714784145355], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e5c9786c42f95b676e1ecf861f106cab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.31906890869140625], [0.04625708982348442], [0.439250111579895], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_1ec8786a9b522fabfba0c589acc22c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16905468702316284], [0.07408490031957626], [0.13317765295505524], [0.36288416385650635], [0.0633009746670723]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9b1ab8f3ba30ad25e08c7ec94d2d5608(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15759168565273285], [0.09588633477687836], [0.37041568756103516], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.31906890869140625], [0.02482573315501213], [0.439250111579895], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_cbf3f029c7bebc4af2860099508d2b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.3282714784145355], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.19767306745052338], [0.23302686214447021], [0.31559082865715027], [0.44974565505981445], [0.12936343252658844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f6e412b64b1321370f360befb81fe668(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.149748757481575], [0.15492382645606995]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.04625708982348442], [0.1838442087173462], [0.45197582244873047], [0.287341833114624]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_409056cbe015232ad1fddd5e03416095(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.005169573239982128], [0.06563594937324524], [0.019044138491153717], [0.1293209046125412], [-0.01654825359582901]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_ad2aadbe9531128fa3d9d06b5c4e031b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4030136466026306], [0.48801735043525696], [0.401704341173172], [0.36288416385650635], [0.1866285353899002]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.08599822223186493], [0.21465438604354858], [0.023872841149568558], [0.1336366832256317], [0.12001485377550125]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e5e82d6649e2aa81e24416ca650f8ceb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2669719457626343], [0.3428363800048828], [0.49236804246902466], [0.4901740849018097], [0.2975691258907318]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.22683300077915192], [0.02482573315501213], [0.1838442087173462], [0.08620838075876236], [0.13948924839496613]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_8ecddb585842466f44a022f10ea533f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.012724664062261581], [0.08693233877420425], [0.11657001823186874], [0.09260812401771545], [0.010530282743275166]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.005169573239982128], [0.06563594937324524], [0.019044138491153717], [0.1293209046125412], [-0.01654825359582901]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_3289c1cdb1e147626c084ecff5033a5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.0], [0.0], [0.0], [-0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[1.406264066696167], [0.24497660994529724], [0.8366292119026184], [-0.3964315354824066], [2.5714917182922363]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_3d584be8881a885f075443f30c649551(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ad23cbc83076076e512c9c188c78835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_3d584be8881a885f075443f30c649551(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_408244feb35de9a39cd8dcd5d3edc17c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_778c86ff8c661fdc9d25c00052811c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_408244feb35de9a39cd8dcd5d3edc17c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_209f79aced5bf85d117606909816fdd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3ef633fe488b4250962312511a53eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_209f79aced5bf85d117606909816fdd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_da573b992cb67d478a1f365d9f90c432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.34141385555267334, 0.08070738613605499, 0.03860168904066086, 0.49608147144317627, 0.41707998514175415, 0.07405710965394974, 0.39049777388572693, 0.14312376081943512, 0.4575801193714142, 0.44072431325912476, 0.48256993293762207, 0.10408809781074524, 0.3957173824310303, 0.045056845992803574, 0.2210846096277237, 0.11286043375730515, 0.109720878303051, 0.17332060635089874, 0.47122129797935486, 0.48317795991897583], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_babe53a7f60a08fa112856b8eab6ec44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.34141385555267334, 0.08070738613605499, 0.03860168904066086, 0.49608147144317627, 0.41707998514175415, 0.07405710965394974, 0.39049777388572693, 0.14312376081943512, 0.4575801193714142, 0.44072431325912476, 0.48256993293762207, 0.10408809781074524, 0.3957173824310303, 0.045056845992803574, 0.2210846096277237, 0.11286043375730515, 0.109720878303051, 0.17332060635089874, 0.47122129797935486, 0.48317795991897583], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_ceb34b6ab5cd4c0956d4fd4f79209f6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14522317051887512], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_bc9b89520247c368be95c21d5048165a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.440621554851532], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.32093942165374756], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7fc4d437235ad0c98453977dff3950cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.2501605749130249], [0.0006863751332275569], [0.08632545918226242]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_0f4413c5b528c6956fc773e5c0432c13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16857455670833588], [0.48723578453063965], [0.0007394644781015813], [0.12745331227779388]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.32093942165374756], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_c3997395517807b014087c67224b8aee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14522317051887512], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.40112707018852234], [0.16776657104492188], [0.05913810804486275], [0.45849156379699707]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_8c284825fd4168f064b8502e78587e4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28212907910346985], [0.440621554851532], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.16157600283622742], [0.17247040569782257], [0.42010706663131714]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_2260b4770e4a96c82251955f2a0f1d6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.04221170023083687], [0.1649954915046692], [-0.0011601648293435574], [-0.019290968775749207]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.02299167960882187], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_0bd096cf33b73ba8b7fee739793e38ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49404314160346985], [0.43382689356803894], [0.1448354572057724], [0.3873145282268524]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.33240923285484314], [0.013569344766438007], [0.04725205898284912], [0.23816898465156555]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_cf4680715055edbf1d218d60ba8f1c29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28212907910346985], [0.48723578453063965], [0.13090020418167114], [0.48451167345046997]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.21341432631015778], [0.10365208238363266], [0.052328769117593765], [0.030598077923059464]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_62e79b6ec3762afef49d29166928ca15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.011106634512543678], [0.16120393574237823], [0.0076672681607306], [0.06769919395446777]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.04221170023083687], [0.14200380444526672], [-0.0011601647129282355], [-0.019290968775749207]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_dcaf0837bc8e85995d5b0979e349a080(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.16190889477729797], [-0.0], [-0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[4.800584316253662], [0.11910460889339447], [1.1513140201568604], [1.2849512100219727]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f866141f6138d4c2131ae7ec085de7de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72710e2413ba48d90b8e3a50b3b85b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08dce775b2f3177e16eacc4467d60d8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_72710e2413ba48d90b8e3a50b3b85b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a4111e1475b599fa44744ebe9879d21b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18006691336631775, 0.47249430418014526, 0.2704814374446869, 0.28356489539146423], [0.1444801688194275, 0.26884371042251587, 0.32883790135383606, 0.40756726264953613], [0.40327703952789307, 0.16330288350582123, 0.4570107161998749, 0.3071627914905548], [0.40327703952789307, 0.16330288350582123, 0.4570107161998749, 0.3071627914905548], [0.4027557671070099, 0.4091954827308655, 0.3253519833087921, 0.04934440553188324]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.18216904997825623, 0.3664987087249756, 0.09633531421422958, 0.16741898655891418], [0.432188481092453, 0.08552742004394531, 0.4801865220069885, 0.10212612897157669], [0.09275709837675095, 0.3269622027873993, 0.2648838758468628, 0.37689346075057983], [0.09275709837675095, 0.3269622027873993, 0.2648838758468628, 0.37689346075057983], [0.1461302638053894, 0.10133280605077744, 0.3515809178352356, 0.4469776153564453]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_b71b31a1f15cafc5c03472a22288aebd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37395c933ecd207df93554234fbdbbd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b71b31a1f15cafc5c03472a22288aebd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d24d2e299067f0114ab676603b7437a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07471240311861038, 0.1784989833831787, 0.0961991548538208, 0.2725340723991394], [0.4185662865638733, 0.41786858439445496, 0.3821410536766052, 0.3345850706100464], [0.39434146881103516, 0.07983526587486267, 0.014400581829249859, 0.29254430532455444], [0.07471240311861038, 0.1784989833831787, 0.0961991548538208, 0.2725340723991394], [0.348670095205307, 0.276864618062973, 0.2532390356063843, 0.433200865983963], [0.07939526438713074, 0.25467807054519653, 0.49734240770339966, 0.2597086429595947], [0.348670095205307, 0.276864618062973, 0.2532390356063843, 0.433200865983963]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.1009109765291214, 0.05554174259305, 0.07083551585674286, 0.32371416687965393], [0.15809713304042816, 0.25184744596481323, 0.13013799488544464, 0.4204864799976349], [0.12200950086116791, 0.45094767212867737, 0.051762133836746216, 0.018873345106840134], [0.1009109765291214, 0.05554174259305, 0.07083551585674286, 0.32371416687965393], [0.3411698639392853, 0.017546646296977997, 0.26674339175224304, 0.05254773423075676], [0.46994319558143616, 0.45695093274116516, 0.27365565299987793, 0.011935419403016567], [0.3411698639392853, 0.017546646296977997, 0.26674339175224304, 0.05254773423075676]], dtype='float32').reshape([7, 4]),
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