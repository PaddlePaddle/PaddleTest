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


    class TestPrimitiveOp_1edf3532ee2c65867b2ab6dbbb775a2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.1631958782672882]], [[0.4830591678619385]], [[0.13640281558036804]], [[0.06289021670818329]], [[0.3381606638431549]], [[0.1730637103319168]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6873317360877991]], [[0.5819525718688965]], [[0.7999815344810486]], [[0.5466142296791077]], [[0.614906907081604]], [[0.6087080240249634]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_9c33ec4410c114c7c46c7c4d41125c58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.16262555122375488]], [[0.4710995554924011]], [[0.46580007672309875]], [[0.40595364570617676]], [[0.4132225513458252]], [[0.436888724565506]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6331064105033875]], [[0.6973821520805359]], [[0.7257800102233887]], [[0.5502040386199951]], [[0.7628018856048584]], [[0.7564184069633484]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_ae3c7edd9669d3e65a2deb56f1a0d5ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.44883012771606445, 0.1614665985107422]], [[0.4670383036136627, 0.20265744626522064]], [[0.20590291917324066, 0.1419561803340912]], [[0.24326932430267334, 0.28298479318618774]], [[0.3838028907775879, 0.49517402052879333]], [[0.48496755957603455, 0.13528983294963837]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.3943357467651367, 0.0671614482998848]], [[0.47695010900497437, 0.19709175825119019]], [[0.0027434383518993855, 0.3336830139160156]], [[0.2079385668039322, 0.07217258960008621]], [[0.1960109919309616, 0.05200967937707901]], [[0.20042534172534943, 0.4425562620162964]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_faa1d929f4ad8bde648d254fadf85033(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.08073487877845764, 0.32428503036499023]], [[0.3015163540840149, 0.4314993619918823]], [[0.16413716971874237, 0.15354189276695251]], [[0.07749584317207336, 0.1363573968410492]], [[0.16635501384735107, 0.1668197214603424]], [[0.08065397292375565, 0.22361506521701813]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.3943357467651367, 0.0671614482998848]], [[0.47695010900497437, 0.19709175825119019]], [[0.0027434383518993855, 0.3336830139160156]], [[0.2079385668039322, 0.07217258960008621]], [[0.1960109919309616, 0.05200967937707901]], [[0.20042534172534943, 0.4425562620162964]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_dd57442b44ce28029c6d976c5d2a3165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ae748e3ee78da0b91c7817413d3dc0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.16726118326187134, 0.34479621052742004]], [[0.3814340829849243, 0.18334932625293732]], [[0.3706839680671692, 0.2882494330406189]], [[0.20342600345611572, 0.3141588270664215]], [[0.40584495663642883, 0.05891812965273857]], [[0.4210466742515564, 0.15735946595668793]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_b2ff266e9919e71a82daf03a57bfd65c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.05183764174580574, 0.1322854906320572, 0.3363713026046753, 0.13401319086551666, 0.47789466381073, 0.2742125988006592, 0.16450469195842743, 0.27145132422447205, 0.2347048819065094, 0.46265271306037903, 0.4377806782722473, 0.1997579038143158, 0.23424941301345825, 0.18413367867469788, 0.47968658804893494, 0.4680825173854828], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_76e3726c975db230f3b6fa73a4aeb8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.05183764174580574, 0.1322854906320572, 0.3363713026046753, 0.13401319086551666, 0.47789466381073, 0.2742125988006592, 0.16450469195842743, 0.27145132422447205, 0.2347048819065094, 0.46265271306037903, 0.4377806782722473, 0.1997579038143158, 0.23424941301345825, 0.18413367867469788, 0.47968658804893494, 0.4680825173854828], dtype='float32').reshape([16]),
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


    class TestPrimitiveOp_fac169da6318869ac0883995d04f0642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb895c87059cbf742283fa2aede38721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_fac169da6318869ac0883995d04f0642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01fbc341750f8c685ca87b800f5f64f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03942854702472687, 0.47896647453308105, 0.0707368552684784, 0.15708090364933014], [0.34924668073654175, 0.4959641695022583, 0.2033059000968933, 0.2109263688325882], [0.31559088826179504, 0.3385964035987854, 0.3923597037792206, 0.36877623200416565], [0.33733460307121277, 0.14558610320091248, 0.30153128504753113, 0.06012469157576561], [0.35864126682281494, 0.227564737200737, 0.30506813526153564, 0.17007820308208466]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.09926425665616989, 0.23578712344169617, 0.13742288947105408, 0.06377401947975159], [0.37992334365844727, 0.08063960075378418, 0.30269384384155273, 0.11003783345222473], [0.35751020908355713, 0.22279569506645203, 0.25694164633750916, 0.4777662456035614], [0.13734689354896545, 0.49705782532691956, 0.2898792028427124, 0.38485613465309143], [0.254555881023407, 0.04162680357694626, 0.20142796635627747, 0.09924576431512833]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_2352dfd20599fa40540a807fa1dfb7fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28719210624694824, 0.23055420815944672, 0.04956628754734993, 0.22986920177936554], [0.29731088876724243, 0.4535277485847473, 0.4582275152206421, 0.4929276704788208], [0.4882139265537262, 0.21820244193077087, 0.09530343115329742, 0.26205146312713623], [0.29731088876724243, 0.4535277485847473, 0.4582275152206421, 0.4929276704788208], [0.4882139265537262, 0.21820244193077087, 0.09530343115329742, 0.26205146312713623]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.3666442632675171, 0.038568176329135895, 0.14608372747898102, 0.28126558661460876], [0.43499240279197693, 0.4896965026855469, 0.4885249137878418, 0.4071301221847534], [0.42328980565071106, 0.2418367564678192, 0.06388422101736069, 0.4153708517551422], [0.43499240279197693, 0.4896965026855469, 0.4885249137878418, 0.4071301221847534], [0.42328980565071106, 0.2418367564678192, 0.06388422101736069, 0.4153708517551422]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_a05ff82779dd26852f80166db83b3eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.046371277421712875], [0.0077659436501562595], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.047508757561445236], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.18618053197860718], [0.042280279099941254], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_b6c13f95ebfa7cff8df687a2cb70894d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42142167687416077], [0.338043212890625], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.2514757812023163], [0.4964841306209564], [0.21071745455265045], [0.1466875672340393], [0.4495491087436676], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_4351eea4ee042ad2a2a3c6a344c4b021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.36908185482025146], [0.2961925268173218], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.06219051405787468], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.18618053197860718], [0.007991598919034004], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_d19d9aa3fa954ae2d05d91586a20097e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.338043212890625], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.2514757812023163], [0.4430234134197235], [0.21071745455265045], [0.002510953461751342], [0.4495491087436676], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_405078fc70779c93cef045fc62ee2003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4386983811855316], [0.046371277421712875], [0.0077659436501562595], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.047508757561445236], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.042280279099941254], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_57e7727d5c393d3e0fea3d8ef867acbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42142167687416077], [0.39432990550994873], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.11843360215425491], [0.4964841306209564], [0.09898043423891068], [0.1466875672340393], [0.21377909183502197], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e910b9418427bd8fc6696b702110f831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06809649616479874], [0.1168467253446579], [0.05836332216858864], [0.04395139962434769], [-0.006267378106713295], [0.004919853061437607], [-0.005880832672119141], [-0.006536051165312529], [0.1567060500383377]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.012420357204973698], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_ca1fd952653f8ac34098bb032c9f47b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4386983811855316], [0.36908185482025146], [0.2961925268173218], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.06219051405787468], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.007991598919034004], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_08d48b712c50a1eefd14e30f5fd6d524(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.39432990550994873], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.11843360215425491], [0.4430234134197235], [0.09898043423891068], [0.002510953461751342], [0.21377909183502197], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9e498f62dda68437b25faae6a928a937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09032297879457474], [0.13734589517116547], [0.08073534816503525], [0.004634413868188858], [0.11631671339273453], [0.03611272946000099], [0.004578196443617344], [-0.0018971551908180118], [0.027492618188261986]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.055676139891147614], [0.1168467253446579], [0.05836332216858864], [0.04395139962434769], [-0.006267378106713295], [0.004919853061437607], [-0.005880832672119141], [-0.006536051165312529], [0.1567060500383377]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_93fac07e4e4a18e6c488efb731c340cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22308222949504852], [0.0], [0.0], [0.0], [-0.0], [0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3835883140563965], [0.14925214648246765], [0.2771032452583313], [-8.483701705932617], [1.0538820028305054], [0.8637640476226807], [2.2845304012298584], [-2.4451851844787598], [-4.699932098388672]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_1af58d553c5e7b6ec645021f13cf8a4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_255402753063472a8b72eceb8247eefe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.24009037017822266]], [[0.36598658561706543]], [[0.45717522501945496]], [[0.01734936609864235]], [[0.4299929738044739]], [[0.46376073360443115]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6013274788856506]], [[0.5645684599876404]], [[0.6824908256530762]], [[0.7009567022323608]], [[0.7100949287414551]], [[0.6275843977928162]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_9989a99a60665a5fe03f40208d83bb1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4855091869831085]], [[0.48198366165161133]], [[0.10854921489953995]], [[0.05358770489692688]], [[0.39330196380615234]], [[0.21844549477100372]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6730449199676514]], [[0.807695209980011]], [[0.7492591738700867]], [[0.6105006337165833]], [[0.7242305874824524]], [[0.5755680203437805]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_886d9b4429a18d267d5271361b85d520(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae65b74a3d2a7e76a0b687888f5df363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_886d9b4429a18d267d5271361b85d520(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbf848776a40e812e4866203f4f9150f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22895076870918274, 0.43839433789253235, 0.09775096923112869, 0.4802975058555603], [0.3041378855705261, 0.25429245829582214, 0.43246856331825256, 0.3057610094547272], [0.4121575355529785, 0.4209367334842682, 0.025442074984312057, 0.0592651404440403], [0.3041378855705261, 0.25429245829582214, 0.43246856331825256, 0.3057610094547272], [0.4121575355529785, 0.4209367334842682, 0.025442074984312057, 0.0592651404440403], [0.286880224943161, 0.007150974590331316, 0.1812123954296112, 0.08457574993371964], [0.286880224943161, 0.007150974590331316, 0.1812123954296112, 0.08457574993371964]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.12005609273910522, 0.18032310903072357, 0.24188463389873505, 0.0608525313436985], [0.003236803924664855, 0.434544175863266, 0.31264448165893555, 0.36690962314605713], [0.04446536302566528, 0.2904546856880188, 0.4626550078392029, 0.05316312983632088], [0.003236803924664855, 0.434544175863266, 0.31264448165893555, 0.36690962314605713], [0.04446536302566528, 0.2904546856880188, 0.4626550078392029, 0.05316312983632088], [0.1808030754327774, 0.28809061646461487, 0.29978424310684204, 0.401498407125473], [0.1808030754327774, 0.28809061646461487, 0.29978424310684204, 0.401498407125473]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_a6b307b98a81c3cb5e2d3782c9c0a038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.04505692422389984, 0.4173981845378876, 0.0506727397441864, 0.004974461626261473, 0.0021464754827320576], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e7370d269344c78293369d886645c1aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.27147242426872253, 0.34147074818611145, 0.1349279284477234, 0.04442029446363449, 0.05202075466513634, 0.2485688030719757], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b0dde865db2b12ffda7eb548d45e24a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.3307744562625885, 0.4149071276187897, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c49ab63c0f37c7c1c395ff0f954a60d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.4658578038215637, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4fb159eb90dbe4315554fe059c8cf65b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1420275717973709, 0.3307744562625885, 0.4149071276187897, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4ecc6434e46daa88692ec15cd9bcf5a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.4658578038215637, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_aebd32b73ee0afab73056eed93688b63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_baadb69fa6bb53d57921a039bd2fecaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_61f9a57961b58eb0fd23b597f68cf79c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.04189087823033333, 0.03793960437178612, -0.019696950912475586, -0.00199795956723392, -0.005094417370855808, -0.05107930675148964], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e61deb9c76ef61c92651abddcbccffbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23392561078071594, 0.09065317362546921, 0.33928659558296204, 0.11514031887054443, 0.14891447126865387, 0.22644412517547607], dtype='float32').reshape([6]),
                paddle.to_tensor([0.19883015751838684, 0.4130263328552246, 0.302883505821228, 0.13395777344703674, 0.13623738288879395, 0.2516404986381531], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e57ba4488d9926ef9ac3d6b03bde82cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3495340347290039, 0.3608149290084839, 0.2594084143638611, 0.25006911158561707, 0.2193455547094345, 0.35870659351348877], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3054583668708801, 0.21985271573066711, 0.06001290678977966, 0.2806971073150635, 0.21360710263252258, 0.2312326282262802], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2b44e8c23602aa897759c3b74822477f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.13624942302703857, 0.2611750066280365, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a95bb0e6476e72a1c45be4c304f3763d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.45571792125701904, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_3d382a66129aed15889077206014316e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.149177074432373, 0.6195820569992065, -1.1968660354614258, -0.014567945152521133, -1.4922434091567993, -0.3225652575492859], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.8666291832923889, 1.1695619821548462, -0.5603955984115601, 0.3037809133529663, 0.7104108333587646, 1.1143471002578735], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_56cbafa80489549d60d83fd495911b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4080923febfddac47c2c2c1160b863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_56cbafa80489549d60d83fd495911b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_ec72a5adb5bb15bb7b0443e0b5bd5772(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.06644057482481003, 0.0018614486325532198, 0.21422740817070007, 0.006235000677406788, 0.3750665485858917, 0.19984327256679535, 0.4298643171787262, 0.3690134882926941, 0.419566810131073, 0.12179119884967804, 0.4229738712310791, 0.4515219032764435, 0.41391289234161377, 0.24803227186203003, 0.41406798362731934, 0.24153155088424683, 0.07568644732236862, 0.3699430823326111, 0.23401470482349396, 0.25811436772346497, 0.40426042675971985, 0.43211135268211365, 0.1867329478263855, 0.0654502809047699], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_b9456f11f6dcdf1b7f424725ab1046e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06644057482481003, 0.0018614486325532198, 0.21422740817070007, 0.006235000677406788, 0.3750665485858917, 0.19984327256679535, 0.4298643171787262, 0.3690134882926941, 0.419566810131073, 0.12179119884967804, 0.4229738712310791, 0.4515219032764435, 0.41391289234161377, 0.24803227186203003, 0.41406798362731934, 0.24153155088424683, 0.07568644732236862, 0.3699430823326111, 0.23401470482349396, 0.25811436772346497, 0.40426042675971985, 0.43211135268211365, 0.1867329478263855, 0.0654502809047699], dtype='float32').reshape([24]),
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


    class TestPrimitiveOp_0818646df908cb2f7c854b7b8891a91a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c1a3395b6ad4d3bf6b4d4e1216ed157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0818646df908cb2f7c854b7b8891a91a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_995dfbca5b61b5719434a1503e104f85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.3004399240016937, 0.37846723198890686, 0.201074481010437, 0.014919416047632694], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_01119198d954be80f32e19b2c7b28ca0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3004399240016937, 0.37846723198890686, 0.201074481010437, 0.014919416047632694], dtype='float32').reshape([4]),
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


    class TestPrimitiveOp_64170e6c21b3a4548689c466f3dc8d6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4689941704273224, 0.284049928188324, 0.1052517294883728, 0.053699977695941925], [0.3941596746444702, 0.1582547277212143, 0.39884933829307556, 0.03849705308675766], [0.13040339946746826, 0.043503545224666595, 0.043750714510679245, 0.13780781626701355], [0.33933836221694946, 0.1820559799671173, 0.24868448078632355, 0.006877193693071604], [0.33933836221694946, 0.1820559799671173, 0.24868448078632355, 0.006877193693071604], [0.13040339946746826, 0.043503545224666595, 0.043750714510679245, 0.13780781626701355]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.2394210398197174, 0.4624246060848236, 0.0883968323469162, 0.46321913599967957], [0.36362117528915405, 0.18431217968463898, 0.3945719599723816, 0.13577550649642944], [0.3852720856666565, 0.3568287193775177, 0.2815832197666168, 0.3905746340751648], [0.185911163687706, 0.23607124388217926, 0.44473662972450256, 0.088658407330513], [0.185911163687706, 0.23607124388217926, 0.44473662972450256, 0.088658407330513], [0.3852720856666565, 0.3568287193775177, 0.2815832197666168, 0.3905746340751648]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_ad1d365b1d764be9549fa58b020d9883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3586193919181824, 0.13367393612861633, 0.07783482223749161, 0.09604031592607498], [0.2363462895154953, 0.015636775642633438, 0.05108630657196045, 0.4619491994380951], [0.19445933401584625, 0.4037427604198456, 0.3171370029449463, 0.12906880676746368], [0.465982049703598, 0.4749557673931122, 0.1601065844297409, 0.08161168545484543], [0.3586193919181824, 0.13367393612861633, 0.07783482223749161, 0.09604031592607498]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.14423157274723053, 0.059380531311035156, 0.3358438014984131, 0.3124944567680359], [0.3744156062602997, 0.44170665740966797, 0.051465265452861786, 0.0004932225565426052], [0.07123012840747833, 0.2832140326499939, 0.02902403473854065, 0.33228036761283875], [0.4094831347465515, 0.4135250747203827, 0.021371887996792793, 0.3999515473842621], [0.14423157274723053, 0.059380531311035156, 0.3358438014984131, 0.3124944567680359]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_73ece6d168a08bd3fa887a7c95aa72af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5ccf27dca01f6f43bde49c34f209a36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b9d10b48039d174609285673c7470fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ec593f2bce8cc31cc74bc1bbe4979185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2bc82ddbac90a0198fe03105a23d1401(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_46890465cca594d33c601a0282add8a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b9d10b48039d174609285673c7470fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_f98357e61dd70738828421f9496e218b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.002530277008190751]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_00c126ef2f810ac5c2bb49e2fb7daaff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2bc82ddbac90a0198fe03105a23d1401(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_fda38fb673d7013a8621f54eb031b22e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16185759007930756]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.002530277008190751]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_04c438cf45bfb6c7c79b1c977637c941(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.9843672513961792]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_db506fcf3e195100965400c62da71968(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.13643445074558258], [0.41118323802948], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_5320bc9d7c44efd1c9ba63ef5a8b1854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.09585190564393997], [0.27821820974349976], [0.4414989948272705], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.42264029383659363], [0.4681013226509094], [0.35276615619659424], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_801b28f8820e614555af8ebe7540acbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.13643445074558258], [0.08216515928506851], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8fab3dfe33cbed4d49dfe6d6819807be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.09585190564393997], [0.27821820974349976], [0.4517127275466919], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.42264029383659363], [0.03228634223341942], [0.35276615619659424], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_08bcd1acb1d6c0a796e8710c05fca718(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.41118323802948], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2dc3819f39454f74b30332cf8abd4a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.2810487151145935], [0.3574141561985016], [0.4414989948272705], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.39316296577453613], [0.4681013226509094], [0.0868077278137207], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_c1f889363b829a1de8b66326e814c024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0036731057334691286], [0.035216521471738815], [0.10041127353906631], [-0.07269255071878433], [0.07417459785938263], [-0.0034735659137368202]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.00018990683020092547], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_0921ff70fb6b3c790643c14cee96832e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.08216515928506851], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2b195a8d6e114fb847b209b91929c613(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.2810487151145935], [0.3574141561985016], [0.4517127275466919], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.39316296577453613], [0.03228634223341942], [0.0868077278137207], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_11cd53271498123ded99d49a82559b9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08572027832269669], [0.0022559084463864565], [-0.014358188025653362], [-0.02135562337934971], [0.13585996627807617], [0.035559557378292084]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.0036731057334691286], [0.035216521471738815], [0.10041127353906631], [-0.07269255071878433], [0.07398469001054764], [-0.0034735659137368202]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_38adeac3e64b0d8fc2e155066dfa61fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.0], [0.0], [-0.0], [0.0025668395683169365], [-0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[1.0428498983383179], [-14.610793113708496], [7.993310928344727], [-2.40390682220459], [0.4554342031478882], [1.097683072090149]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8147e6600d4809d0a0b71cfb47f08a2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41194984316825867, 0.1985234022140503, 0.08035333454608917, 0.2548809051513672], [0.389752596616745, 0.49257543683052063, 0.24452143907546997, 0.4366907775402069], [0.4164615869522095, 0.11515739560127258, 0.4882013499736786, 0.2104092687368393], [0.04727732017636299, 0.46199488639831543, 0.14487098157405853, 0.17658351361751556]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.3674372732639313, 0.39325854182243347, 0.44171425700187683, 0.23020479083061218], [0.43334540724754333, 0.047591667622327805, 0.1316508948802948, 0.4297671318054199], [0.15011842548847198, 0.0014065280556678772, 0.15698941051959991, 0.06276851147413254], [0.20743593573570251, 0.3237133324146271, 0.13536261022090912, 0.3818800747394562]], dtype='float32').reshape([4, 4]),
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


    class TestPrimitiveOp_4ce565c9ee4656260e01f51662775504(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3abfa49130a9550c068f1e48964c3bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4ce565c9ee4656260e01f51662775504(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67ee3dd02410dfeec83c2606c1d265ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14792421460151672, 0.060222066938877106, 0.3426845371723175, 0.2729192078113556], [0.14792421460151672, 0.060222066938877106, 0.3426845371723175, 0.2729192078113556], [0.20768477022647858, 0.10328125953674316, 0.14738108217716217, 0.3912164270877838], [0.4820886254310608, 0.09006587415933609, 0.2595599591732025, 0.43800047039985657], [0.45774030685424805, 0.1348615139722824, 0.3947620391845703, 0.23843154311180115], [0.16375482082366943, 0.475871205329895, 0.369335412979126, 0.20670752227306366], [0.354208767414093, 0.34655365347862244, 0.2486346811056137, 0.16356748342514038]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.11575445532798767, 0.016847269609570503, 0.3592566251754761, 0.3711000978946686], [0.11575445532798767, 0.016847269609570503, 0.3592566251754761, 0.3711000978946686], [0.05424733832478523, 0.4543360769748688, 0.3827211558818817, 0.4861321449279785], [0.3619614541530609, 0.17202797532081604, 0.3347017168998718, 0.004203255288302898], [0.39603060483932495, 0.3538707494735718, 0.27019816637039185, 0.40859922766685486], [0.24719764292240143, 0.20523923635482788, 0.3153506815433502, 0.08138352632522583], [0.4453897476196289, 0.10803820192813873, 0.32960209250450134, 0.17188863456249237]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_cb73920e4bf56c662d47981321092e3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1e89e0aab4be788d3b8fe0894876a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_cb73920e4bf56c662d47981321092e3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55af0a7f86d61426d02aa49b57653daf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118b701eb1f6f3557817bdfda5c77263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_55af0a7f86d61426d02aa49b57653daf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a89dfe01d48a19b39a14b931a6f22cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e8f1fce68a2ec0919891c25fad229f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.30025285482406616, 0.27773964405059814, 0.12615980207920074, 0.10717370361089706], [0.35898715257644653, 0.3946774899959564, 0.3250655233860016, 0.29423171281814575], [0.35898715257644653, 0.3946774899959564, 0.3250655233860016, 0.29423171281814575], [0.0367368683218956, 0.11908857524394989, 0.15985457599163055, 0.2605140507221222], [0.30330580472946167, 0.4362056255340576, 0.18896079063415527, 0.44536158442497253], [0.2442024052143097, 0.21146763861179352, 0.25890666246414185, 0.26765894889831543]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.4743831753730774, 0.2996712923049927, 0.05848970636725426, 0.21108603477478027], [0.009142567403614521, 0.35248109698295593, 0.30357590317726135, 0.11087937653064728], [0.009142567403614521, 0.35248109698295593, 0.30357590317726135, 0.11087937653064728], [0.11678148061037064, 0.35593029856681824, 0.24865007400512695, 0.439622163772583], [0.2514708936214447, 0.31273120641708374, 0.049657903611660004, 0.19320592284202576], [0.45483604073524475, 0.08000791817903519, 0.41738295555114746, 0.24432004988193512]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_34430c3fd1a44bdc5a125cfec33c32dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f62c64312ccb8d87c12c185fd5e515e
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.7734333872795105, 2.265752077102661, 0.539688229560852, 6.51957893371582], [2.907357931137085, 0.14618602395057678, 0.6784908771514893, 2.3595669269561768]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_1020999b2935b84f30f370bb1213df46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3db6dde5bf3f37813420aec60ad447b6
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1.3565759658813477, 3.342909336090088, 1.0730841159820557, 0.8684794306755066], [4.103296756744385, 2.149132251739502, 0.5966194868087769, 0.18993321061134338]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_bfca6e46c56e8ef23f6514ee0a4a778f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.3625732958316803], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f8eb4ee11549035ab8c380c6ad29b613(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26832517981529236], [0.16028515994548798], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b4ab1f85ce1d6f1a2325916a9666b51b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.33523043990135193], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9c014d1c45920997863232e2eab02a78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.16028515994548798], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.40533360838890076], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e6b51e073612bfe31c9d4f40522c6e9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.3625732958316803], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_87583d0e13c76ec0db61c846310bb3a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26832517981529236], [0.4581025242805481], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_bafc7ccac5ffada2f69f8a80cf0cdf9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06795760244131088], [0.04772914946079254], [-0.06907474249601364], [-0.05889728665351868], [0.017041902989149094]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e16a601d4362d92fc7e6825d329d74cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.33523043990135193], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_cc2bed9235e217efb3e6b5908f487b10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.4581025242805481], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.40533360838890076], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_12dbf1b47e85ba1c9d96839c9e1fb377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.03523365408182144], [0.05821434408426285], [0.019736213609576225], [-0.04269413650035858], [0.0002444145502522588]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.06795760244131088], [0.04772914946079254], [-0.06907474249601364], [-0.05889728665351868], [0.017041902989149094]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_298ac4f8b2bff361d07e93651e119a67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.9287696480751038], [0.18011359870433807], [4.499898433685303], [-0.37951698899269104], [-68.72540283203125]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_26044891a88920bf631dbe9130e8cde5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc018957f9090c6abad90442ef0f1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_26044891a88920bf631dbe9130e8cde5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1512ed0ecb206cbd09e302ab4076bf02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_307fd46db83fe8424d191af1c7b27e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_1512ed0ecb206cbd09e302ab4076bf02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a1f319192b99955f98ace9f4f0a590e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbee9d2b238adc567e6ffb01377bc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2a1f319192b99955f98ace9f4f0a590e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f74a83ba56eb3141df21b59e1afd71f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.26205939054489136, 0.14697584509849548, 0.4793679714202881, 0.02872663363814354, 0.21173322200775146, 0.23947587609291077, 0.37367042899131775, 0.32054823637008667, 0.2866531014442444, 0.07623961567878723, 0.13533982634544373, 0.0694308876991272, 0.11348035931587219, 0.21457159519195557, 0.39497604966163635, 0.06678082793951035, 0.23896890878677368, 0.24635815620422363, 0.29478490352630615, 0.0675901472568512], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_53675dea3dfd1a1a97e850b516cd6de6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.26205939054489136, 0.14697584509849548, 0.4793679714202881, 0.02872663363814354, 0.21173322200775146, 0.23947587609291077, 0.37367042899131775, 0.32054823637008667, 0.2866531014442444, 0.07623961567878723, 0.13533982634544373, 0.0694308876991272, 0.11348035931587219, 0.21457159519195557, 0.39497604966163635, 0.06678082793951035, 0.23896890878677368, 0.24635815620422363, 0.29478490352630615, 0.0675901472568512], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_4708a4529cb2268842a29b94f60f078f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09640353918075562], [0.021595124155282974], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41484373807907104], [0.09736814349889755], [0.3636853098869324], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6da68b18e8b546d16b1c3af32656717f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.061657778918743134], [0.08861131221055984], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9b76c1a0aa7efaae790812370b6bda8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.021595124155282974], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41484373807907104], [0.09475763142108917], [0.012042115442454815], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_0d269b7ee0bc6dccb608d42dc5fc5585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.3512554168701172], [0.08861131221055984], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_89968f33d976645fc0ef466e8f763000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09640353918075562], [0.25064805150032043], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09736814349889755], [0.3636853098869324], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_bf4ab664171f3b354a940b5da1a46368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22977420687675476], [0.061657778918743134], [0.2658151686191559], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_042c3df5367c540ce9a8da57e6f2c2c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09553957730531693], [-0.059163011610507965], [-0.05419258028268814], [0.01992667280137539]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_403a032a88ab02c58b561da2046c0c1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.25064805150032043], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09475763142108917], [0.012042115442454815], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_647fda7b8ecff27ca1d55bb7c59f4560(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22977420687675476], [0.3512554168701172], [0.2658151686191559], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a8d976912321c5e2e5dc990f4e355965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.012408342212438583], [0.04858570545911789], [-0.009754105471074581], [0.04100790247321129]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09553957730531693], [-0.059163011610507965], [-0.05419258028268814], [0.01992667280137539]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_195a934eec21889b667d7c7ad89dcb38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-6.699625015258789], [2.2177040576934814], [-4.555873870849609], [0.5140772461891174]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f866141f6138d4c2131ae7ec085de7de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27127e6315c7f9c65e573fa3118c9b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ffc99c802fb8013e74400cddb6d914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_27127e6315c7f9c65e573fa3118c9b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_cfbcd12182d5b113a5a203da5c4cd018(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40423333644866943, 0.28136950731277466, 0.12250851094722748, 0.0002954275696538389], [0.11568618565797806, 0.27895891666412354, 0.07727570831775665, 0.30709120631217957], [0.033719103783369064, 0.21601463854312897, 0.21620410680770874, 0.25814059376716614], [0.033719103783369064, 0.21601463854312897, 0.21620410680770874, 0.25814059376716614], [0.3263203501701355, 0.06131473928689957, 0.4514027237892151, 0.009571080096065998]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.03664267063140869, 0.14573366940021515, 0.043165650218725204, 0.07306309789419174], [0.4919965863227844, 0.31887757778167725, 0.07879022508859634, 0.3835740387439728], [0.11035966128110886, 0.15233902633190155, 0.18884484469890594, 0.39235854148864746], [0.11035966128110886, 0.15233902633190155, 0.18884484469890594, 0.39235854148864746], [0.06388995051383972, 0.09963720291852951, 0.2488420158624649, 0.4360834062099457]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_bd837fa977b8129317dea4f35c0fad8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1972e241ed8bc38a9fec772f00b9e9d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4adb159a881a57198709244f578ad58
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_bd837fa977b8129317dea4f35c0fad8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463eff677ea0c677a6bb1167e1fc2638
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86525f90cb9182514e4dc4fdf6216bc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16710375249385834, 0.3140609860420227, 0.03993195667862892, 0.4403045177459717], [0.18602943420410156, 0.47431832551956177, 0.39921894669532776, 0.28117549419403076], [0.026671169325709343, 0.3489063084125519, 0.31338927149772644, 0.1287972778081894], [0.16710375249385834, 0.3140609860420227, 0.03993195667862892, 0.4403045177459717], [0.45457276701927185, 0.10955125093460083, 0.07635635882616043, 0.00366982095874846], [0.08987396210432053, 0.48809531331062317, 0.3292236626148224, 0.40736350417137146], [0.45457276701927185, 0.10955125093460083, 0.07635635882616043, 0.00366982095874846]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.3205306828022003, 0.15136130154132843, 0.20435881614685059, 0.08078902214765549], [0.27505624294281006, 0.47417762875556946, 0.1448349505662918, 0.062408119440078735], [0.31028637290000916, 0.10561846196651459, 0.38038167357444763, 0.11800261586904526], [0.3205306828022003, 0.15136130154132843, 0.20435881614685059, 0.08078902214765549], [0.11132610589265823, 0.044622235000133514, 0.33257749676704407, 0.20738846063613892], [0.48840150237083435, 0.40047183632850647, 0.256814569234848, 0.2609950304031372], [0.11132610589265823, 0.044622235000133514, 0.33257749676704407, 0.20738846063613892]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_fbff4c7f33805541db1925229045cfe0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.1631958782672882]], [[0.4830591678619385]], [[0.13640281558036804]], [[0.06289021670818329]], [[0.3381606638431549]], [[0.1730637103319168]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6873317360877991]], [[0.5819525718688965]], [[0.7999815344810486]], [[0.5466142296791077]], [[0.614906907081604]], [[0.6087080240249634]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_08c8b792859e1576508d83d3aeed50ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.16262555122375488]], [[0.4710995554924011]], [[0.46580007672309875]], [[0.40595364570617676]], [[0.4132225513458252]], [[0.436888724565506]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6331064105033875]], [[0.6973821520805359]], [[0.7257800102233887]], [[0.5502040386199951]], [[0.7628018856048584]], [[0.7564184069633484]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_352b0f8ec4c9f92b777107d8237458c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_758bf3801ed88050004599925b40e60d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.44883012771606445, 0.1614665985107422]], [[0.4670383036136627, 0.20265744626522064]], [[0.20590291917324066, 0.1419561803340912]], [[0.24326932430267334, 0.28298479318618774]], [[0.3838028907775879, 0.49517402052879333]], [[0.48496755957603455, 0.13528983294963837]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.3943357467651367, 0.0671614482998848]], [[0.47695010900497437, 0.19709175825119019]], [[0.0027434383518993855, 0.3336830139160156]], [[0.2079385668039322, 0.07217258960008621]], [[0.1960109919309616, 0.05200967937707901]], [[0.20042534172534943, 0.4425562620162964]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_2352362f3ad9893ccad51e7a3e8562f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_758bf3801ed88050004599925b40e60d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.08073487877845764, 0.32428503036499023]], [[0.3015163540840149, 0.4314993619918823]], [[0.16413716971874237, 0.15354189276695251]], [[0.07749584317207336, 0.1363573968410492]], [[0.16635501384735107, 0.1668197214603424]], [[0.08065397292375565, 0.22361506521701813]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.3943357467651367, 0.0671614482998848]], [[0.47695010900497437, 0.19709175825119019]], [[0.0027434383518993855, 0.3336830139160156]], [[0.2079385668039322, 0.07217258960008621]], [[0.1960109919309616, 0.05200967937707901]], [[0.20042534172534943, 0.4425562620162964]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_8a8f060e670c38a700f91016cbd9e31a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e04add1b8d6435475bb578528599f0c8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.16726118326187134, 0.34479621052742004]], [[0.3814340829849243, 0.18334932625293732]], [[0.3706839680671692, 0.2882494330406189]], [[0.20342600345611572, 0.3141588270664215]], [[0.40584495663642883, 0.05891812965273857]], [[0.4210466742515564, 0.15735946595668793]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_8d2769dfc3949d1e06f287173a2f1daf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ec4a74d6c67bc08e22f55b7a09eba1e
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.05183764174580574, 0.1322854906320572, 0.3363713026046753, 0.13401319086551666, 0.47789466381073, 0.2742125988006592, 0.16450469195842743, 0.27145132422447205, 0.2347048819065094, 0.46265271306037903, 0.4377806782722473, 0.1997579038143158, 0.23424941301345825, 0.18413367867469788, 0.47968658804893494, 0.4680825173854828], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_32399fe0661927e67720b55d1e04531d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ec4a74d6c67bc08e22f55b7a09eba1e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.05183764174580574, 0.1322854906320572, 0.3363713026046753, 0.13401319086551666, 0.47789466381073, 0.2742125988006592, 0.16450469195842743, 0.27145132422447205, 0.2347048819065094, 0.46265271306037903, 0.4377806782722473, 0.1997579038143158, 0.23424941301345825, 0.18413367867469788, 0.47968658804893494, 0.4680825173854828], dtype='float32').reshape([16]),
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


    
    class PrimitiveOp_98a1baa154b55aad6489b624740cbd2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e03089cc3155ce56d1fe53af31713020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98a1baa154b55aad6489b624740cbd2e
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5db47aa1f30c2d1deda3d27c56e72eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61e91baaf82dbe8d75e84cb80836119f
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e03089cc3155ce56d1fe53af31713020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98a1baa154b55aad6489b624740cbd2e
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_de9934199147b3067d2e8d80b93a8a1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03942854702472687, 0.47896647453308105, 0.0707368552684784, 0.15708090364933014], [0.34924668073654175, 0.4959641695022583, 0.2033059000968933, 0.2109263688325882], [0.31559088826179504, 0.3385964035987854, 0.3923597037792206, 0.36877623200416565], [0.33733460307121277, 0.14558610320091248, 0.30153128504753113, 0.06012469157576561], [0.35864126682281494, 0.227564737200737, 0.30506813526153564, 0.17007820308208466]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.09926425665616989, 0.23578712344169617, 0.13742288947105408, 0.06377401947975159], [0.37992334365844727, 0.08063960075378418, 0.30269384384155273, 0.11003783345222473], [0.35751020908355713, 0.22279569506645203, 0.25694164633750916, 0.4777662456035614], [0.13734689354896545, 0.49705782532691956, 0.2898792028427124, 0.38485613465309143], [0.254555881023407, 0.04162680357694626, 0.20142796635627747, 0.09924576431512833]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_57b4620b48383867b8dc37ac2f6725cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28719210624694824, 0.23055420815944672, 0.04956628754734993, 0.22986920177936554], [0.29731088876724243, 0.4535277485847473, 0.4582275152206421, 0.4929276704788208], [0.4882139265537262, 0.21820244193077087, 0.09530343115329742, 0.26205146312713623], [0.29731088876724243, 0.4535277485847473, 0.4582275152206421, 0.4929276704788208], [0.4882139265537262, 0.21820244193077087, 0.09530343115329742, 0.26205146312713623]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.3666442632675171, 0.038568176329135895, 0.14608372747898102, 0.28126558661460876], [0.43499240279197693, 0.4896965026855469, 0.4885249137878418, 0.4071301221847534], [0.42328980565071106, 0.2418367564678192, 0.06388422101736069, 0.4153708517551422], [0.43499240279197693, 0.4896965026855469, 0.4885249137878418, 0.4071301221847534], [0.42328980565071106, 0.2418367564678192, 0.06388422101736069, 0.4153708517551422]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_21a2258636535b6c2847f7e4d50b45c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.046371277421712875], [0.0077659436501562595], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.047508757561445236], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.18618053197860718], [0.042280279099941254], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_d7e95e8655a4047d77d35fc090a03a9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42142167687416077], [0.338043212890625], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.2514757812023163], [0.4964841306209564], [0.21071745455265045], [0.1466875672340393], [0.4495491087436676], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_34d877776a29f66b58b2b42bcdbe0863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.36908185482025146], [0.2961925268173218], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.06219051405787468], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.18618053197860718], [0.007991598919034004], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_ed1b0df3b6a207f5b7cadf5be6022c01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.338043212890625], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.2514757812023163], [0.4430234134197235], [0.21071745455265045], [0.002510953461751342], [0.4495491087436676], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_8c19605fc0fc24710cbac0c7ad1a212d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4386983811855316], [0.046371277421712875], [0.0077659436501562595], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.047508757561445236], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.042280279099941254], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_7341f8f6b71a45e6cf98df8432b22bf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42142167687416077], [0.39432990550994873], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.11843360215425491], [0.4964841306209564], [0.09898043423891068], [0.1466875672340393], [0.21377909183502197], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_d3a376af6af27d8fe73714b0fefd3051(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06809649616479874], [0.1168467253446579], [0.05836332216858864], [0.04395139962434769], [-0.006267378106713295], [0.004919853061437607], [-0.005880832672119141], [-0.006536051165312529], [0.1567060500383377]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.012420357204973698], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_ea585a543b7a9bb73f623bed0cc0b0e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4386983811855316], [0.36908185482025146], [0.2961925268173218], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.06219051405787468], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.007991598919034004], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_c6f744eddb4ec3ffb64fd8ad926b2778(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.39432990550994873], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.11843360215425491], [0.4430234134197235], [0.09898043423891068], [0.002510953461751342], [0.21377909183502197], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_daa452174f679c7410bc721d45481f5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09032297879457474], [0.13734589517116547], [0.08073534816503525], [0.004634413868188858], [0.11631671339273453], [0.03611272946000099], [0.004578196443617344], [-0.0018971551908180118], [0.027492618188261986]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.055676139891147614], [0.1168467253446579], [0.05836332216858864], [0.04395139962434769], [-0.006267378106713295], [0.004919853061437607], [-0.005880832672119141], [-0.006536051165312529], [0.1567060500383377]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_bf06aef83e00687d9614818d08478c2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d629b24dffb314652b35d32950ae5d
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22308222949504852], [0.0], [0.0], [0.0], [-0.0], [0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3835883140563965], [0.14925214648246765], [0.2771032452583313], [-8.483701705932617], [1.0538820028305054], [0.8637640476226807], [2.2845304012298584], [-2.4451851844787598], [-4.699932098388672]], dtype='float32').reshape([9, 1]),
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


    class TestPrimitiveOp_ba55a49ad4302711de7af2ca4ec80b12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.24009037017822266]], [[0.36598658561706543]], [[0.45717522501945496]], [[0.01734936609864235]], [[0.4299929738044739]], [[0.46376073360443115]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6013274788856506]], [[0.5645684599876404]], [[0.6824908256530762]], [[0.7009567022323608]], [[0.7100949287414551]], [[0.6275843977928162]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_3e7e93779549efc40992b985d793c902(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecbc5a555e1932001117bdc5a8eaeb6f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4855091869831085]], [[0.48198366165161133]], [[0.10854921489953995]], [[0.05358770489692688]], [[0.39330196380615234]], [[0.21844549477100372]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6730449199676514]], [[0.807695209980011]], [[0.7492591738700867]], [[0.6105006337165833]], [[0.7242305874824524]], [[0.5755680203437805]]], dtype='float32').reshape([6, 1, 1]),
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


    
    class PrimitiveOp_d38a35b9a992a16eb5acbe7c0905502b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[5454, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0cb9eee53cc66134de42df717b19da5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d38a35b9a992a16eb5acbe7c0905502b
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d6918ec4644348297d7506ecdd0301fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[5454, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e09c3b92cd200628636db75e2e688f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6918ec4644348297d7506ecdd0301fe
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d0cb9eee53cc66134de42df717b19da5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d38a35b9a992a16eb5acbe7c0905502b
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_5d92c6f72680a34c59deb64ad044d7a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd341c26c429be4deceb7b7802859f50
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22895076870918274, 0.43839433789253235, 0.09775096923112869, 0.4802975058555603], [0.3041378855705261, 0.25429245829582214, 0.43246856331825256, 0.3057610094547272], [0.4121575355529785, 0.4209367334842682, 0.025442074984312057, 0.0592651404440403], [0.3041378855705261, 0.25429245829582214, 0.43246856331825256, 0.3057610094547272], [0.4121575355529785, 0.4209367334842682, 0.025442074984312057, 0.0592651404440403], [0.286880224943161, 0.007150974590331316, 0.1812123954296112, 0.08457574993371964], [0.286880224943161, 0.007150974590331316, 0.1812123954296112, 0.08457574993371964]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.12005609273910522, 0.18032310903072357, 0.24188463389873505, 0.0608525313436985], [0.003236803924664855, 0.434544175863266, 0.31264448165893555, 0.36690962314605713], [0.04446536302566528, 0.2904546856880188, 0.4626550078392029, 0.05316312983632088], [0.003236803924664855, 0.434544175863266, 0.31264448165893555, 0.36690962314605713], [0.04446536302566528, 0.2904546856880188, 0.4626550078392029, 0.05316312983632088], [0.1808030754327774, 0.28809061646461487, 0.29978424310684204, 0.401498407125473], [0.1808030754327774, 0.28809061646461487, 0.29978424310684204, 0.401498407125473]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_4b2a80f60c4e3321aab23286a996f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.04505692422389984, 0.4173981845378876, 0.0506727397441864, 0.004974461626261473, 0.0021464754827320576], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_5bb160ba65b2147fbcd2aa7a09cf5586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.27147242426872253, 0.34147074818611145, 0.1349279284477234, 0.04442029446363449, 0.05202075466513634, 0.2485688030719757], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_dd1f6993081d1c82dc45294c088a6adf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.3307744562625885, 0.4149071276187897, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e0aa645a6426406bbf6f0c799c36f7e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.4658578038215637, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_45f1de0cdc771def992e6d93cf69431b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1420275717973709, 0.3307744562625885, 0.4149071276187897, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ea49f3d37bbf2c39b3c7d787a2307782(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.4658578038215637, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2ecb8d5a9140d2f363e886a1f8ebf8f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_fbb986ddd24f28f73bd8e3391931cb95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2e82baf81a020999077b005d97e7fd41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.04189087823033333, 0.03793960437178612, -0.019696950912475586, -0.00199795956723392, -0.005094417370855808, -0.05107930675148964], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_bfa64716d4735138d3ac4de18fd6b0b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23392561078071594, 0.09065317362546921, 0.33928659558296204, 0.11514031887054443, 0.14891447126865387, 0.22644412517547607], dtype='float32').reshape([6]),
                paddle.to_tensor([0.19883015751838684, 0.4130263328552246, 0.302883505821228, 0.13395777344703674, 0.13623738288879395, 0.2516404986381531], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4a9a356584245b2259088ca0832da479(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3495340347290039, 0.3608149290084839, 0.2594084143638611, 0.25006911158561707, 0.2193455547094345, 0.35870659351348877], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3054583668708801, 0.21985271573066711, 0.06001290678977966, 0.2806971073150635, 0.21360710263252258, 0.2312326282262802], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b9eed939afcc67b2bc70b0a4a1d5da74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.13624942302703857, 0.2611750066280365, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_62e9aeb7985ce9919dc3653cf5d76021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.45571792125701904, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_84589a315e70edad7975489b9f8f4fe8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b473ce00db4877225484272820f9f3
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.149177074432373, 0.6195820569992065, -1.1968660354614258, -0.014567945152521133, -1.4922434091567993, -0.3225652575492859], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.8666291832923889, 1.1695619821548462, -0.5603955984115601, 0.3037809133529663, 0.7104108333587646, 1.1143471002578735], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_10f766ee1c34861a47b3c196ceb53b0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1722, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0dd58d91dbedcc3dd3bd263abae154f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_10f766ee1c34861a47b3c196ceb53b0c
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1722, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dceb53aa9ac400267555306332669fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_650e91d85880b2766fe04f1a68adfcc9
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0dd58d91dbedcc3dd3bd263abae154f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_10f766ee1c34861a47b3c196ceb53b0c
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a3db29d7ef29531a22f9165f8dfaa7af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c9932fc45296674f2f74b8b06a5dade
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.06644057482481003, 0.0018614486325532198, 0.21422740817070007, 0.006235000677406788, 0.3750665485858917, 0.19984327256679535, 0.4298643171787262, 0.3690134882926941, 0.419566810131073, 0.12179119884967804, 0.4229738712310791, 0.4515219032764435, 0.41391289234161377, 0.24803227186203003, 0.41406798362731934, 0.24153155088424683, 0.07568644732236862, 0.3699430823326111, 0.23401470482349396, 0.25811436772346497, 0.40426042675971985, 0.43211135268211365, 0.1867329478263855, 0.0654502809047699], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_dfdca9e8c76de4efd278ec5f66f035c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c9932fc45296674f2f74b8b06a5dade
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06644057482481003, 0.0018614486325532198, 0.21422740817070007, 0.006235000677406788, 0.3750665485858917, 0.19984327256679535, 0.4298643171787262, 0.3690134882926941, 0.419566810131073, 0.12179119884967804, 0.4229738712310791, 0.4515219032764435, 0.41391289234161377, 0.24803227186203003, 0.41406798362731934, 0.24153155088424683, 0.07568644732236862, 0.3699430823326111, 0.23401470482349396, 0.25811436772346497, 0.40426042675971985, 0.43211135268211365, 0.1867329478263855, 0.0654502809047699], dtype='float32').reshape([24]),
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


    
    class PrimitiveOp_d138b6e3e1e3ce6cb94c9e1c2d763b50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1518, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c0ffcb7d95cf9d946c2dc0b76ed29af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d138b6e3e1e3ce6cb94c9e1c2d763b50
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1518, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c0e275085aa283eb4eafca7e79db49b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6f6823dbd5aee6610d594542cc7360b
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0c0ffcb7d95cf9d946c2dc0b76ed29af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d138b6e3e1e3ce6cb94c9e1c2d763b50
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_52e950918d87dfeb7ef619d877222303(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb4a5f8d2f9250767a759fceb26fb2b8
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.3004399240016937, 0.37846723198890686, 0.201074481010437, 0.014919416047632694], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_50274a48695769ab46a1c6ec57593ab1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb4a5f8d2f9250767a759fceb26fb2b8
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3004399240016937, 0.37846723198890686, 0.201074481010437, 0.014919416047632694], dtype='float32').reshape([4]),
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


    class TestPrimitiveOp_47b05f0e76d4460f921bee0c2741a0a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e17b6d4406f6436b786e9839fcef53
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4689941704273224, 0.284049928188324, 0.1052517294883728, 0.053699977695941925], [0.3941596746444702, 0.1582547277212143, 0.39884933829307556, 0.03849705308675766], [0.13040339946746826, 0.043503545224666595, 0.043750714510679245, 0.13780781626701355], [0.33933836221694946, 0.1820559799671173, 0.24868448078632355, 0.006877193693071604], [0.33933836221694946, 0.1820559799671173, 0.24868448078632355, 0.006877193693071604], [0.13040339946746826, 0.043503545224666595, 0.043750714510679245, 0.13780781626701355]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.2394210398197174, 0.4624246060848236, 0.0883968323469162, 0.46321913599967957], [0.36362117528915405, 0.18431217968463898, 0.3945719599723816, 0.13577550649642944], [0.3852720856666565, 0.3568287193775177, 0.2815832197666168, 0.3905746340751648], [0.185911163687706, 0.23607124388217926, 0.44473662972450256, 0.088658407330513], [0.185911163687706, 0.23607124388217926, 0.44473662972450256, 0.088658407330513], [0.3852720856666565, 0.3568287193775177, 0.2815832197666168, 0.3905746340751648]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_0016fdf60560ebb8408a1dcac35b3fcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3586193919181824, 0.13367393612861633, 0.07783482223749161, 0.09604031592607498], [0.2363462895154953, 0.015636775642633438, 0.05108630657196045, 0.4619491994380951], [0.19445933401584625, 0.4037427604198456, 0.3171370029449463, 0.12906880676746368], [0.465982049703598, 0.4749557673931122, 0.1601065844297409, 0.08161168545484543], [0.3586193919181824, 0.13367393612861633, 0.07783482223749161, 0.09604031592607498]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.14423157274723053, 0.059380531311035156, 0.3358438014984131, 0.3124944567680359], [0.3744156062602997, 0.44170665740966797, 0.051465265452861786, 0.0004932225565426052], [0.07123012840747833, 0.2832140326499939, 0.02902403473854065, 0.33228036761283875], [0.4094831347465515, 0.4135250747203827, 0.021371887996792793, 0.3999515473842621], [0.14423157274723053, 0.059380531311035156, 0.3358438014984131, 0.3124944567680359]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_8d5c4d578be4cba0240c5c7063eb5f93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_51b141bea09177939804da6e8f303f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_512884b76f2741ee13898477c254670a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_82152e87bf2810fdcd10a1bd3624efa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_05117c5bc5f4ddc13be2cc841d2aa8b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_51b141bea09177939804da6e8f303f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_52ab954af73726cc0d3ecd49974b84ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.002530277008190751]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_838896d2f46b84df5ba0652d9d36699f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_82152e87bf2810fdcd10a1bd3624efa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_3c298e0919bdeb1167ed05fabe5b2cdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16185759007930756]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.002530277008190751]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_5f4cdc3793b7766d0aff2347fcee0821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c55035f7a062fc218d885b1c9b738341
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.9843672513961792]], dtype='float32').reshape([1, 1]),
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


    class TestPrimitiveOp_9d5d68dd3c0a643e7dadf6f326b40ad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.13643445074558258], [0.41118323802948], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d1476bb4c25e1431f007822f484880c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.09585190564393997], [0.27821820974349976], [0.4414989948272705], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.42264029383659363], [0.4681013226509094], [0.35276615619659424], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_92ada1f6359edaa0a77522b6131f635f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.13643445074558258], [0.08216515928506851], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_5a0132fe48c6febb60ed9080b5b2c915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.09585190564393997], [0.27821820974349976], [0.4517127275466919], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.42264029383659363], [0.03228634223341942], [0.35276615619659424], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8f3eb05b56fecfee484b71f33c163029(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.41118323802948], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_f89e423fbf66dab1356f4029f22c66a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.2810487151145935], [0.3574141561985016], [0.4414989948272705], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.39316296577453613], [0.4681013226509094], [0.0868077278137207], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_5d7ad52805a8d2f203d8aec1fb00c822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0036731057334691286], [0.035216521471738815], [0.10041127353906631], [-0.07269255071878433], [0.07417459785938263], [-0.0034735659137368202]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.00018990683020092547], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_372df635eece3a46d5820bd4f74cd37d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.08216515928506851], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_0bf716d777a2627d5b6816f516a5fd5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.2810487151145935], [0.3574141561985016], [0.4517127275466919], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.39316296577453613], [0.03228634223341942], [0.0868077278137207], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_f6cae110d18fe6f47b6b4f1ecfb5ec69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08572027832269669], [0.0022559084463864565], [-0.014358188025653362], [-0.02135562337934971], [0.13585996627807617], [0.035559557378292084]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.0036731057334691286], [0.035216521471738815], [0.10041127353906631], [-0.07269255071878433], [0.07398469001054764], [-0.0034735659137368202]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_18b511d8fb0acf426a00d982e9734d2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1783249d8b2fd3a8e22045537013db81
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.0], [0.0], [-0.0], [0.0025668395683169365], [-0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[1.0428498983383179], [-14.610793113708496], [7.993310928344727], [-2.40390682220459], [0.4554342031478882], [1.097683072090149]], dtype='float32').reshape([6, 1]),
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


    class TestPrimitiveOp_d718e8bb039334748b6d4fd9ca1252e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7d91148da6f4d25d1ed84b8a21b1473
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41194984316825867, 0.1985234022140503, 0.08035333454608917, 0.2548809051513672], [0.389752596616745, 0.49257543683052063, 0.24452143907546997, 0.4366907775402069], [0.4164615869522095, 0.11515739560127258, 0.4882013499736786, 0.2104092687368393], [0.04727732017636299, 0.46199488639831543, 0.14487098157405853, 0.17658351361751556]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.3674372732639313, 0.39325854182243347, 0.44171425700187683, 0.23020479083061218], [0.43334540724754333, 0.047591667622327805, 0.1316508948802948, 0.4297671318054199], [0.15011842548847198, 0.0014065280556678772, 0.15698941051959991, 0.06276851147413254], [0.20743593573570251, 0.3237133324146271, 0.13536261022090912, 0.3818800747394562]], dtype='float32').reshape([4, 4]),
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


    
    class PrimitiveOp_3669e8652e2498122420bc5f8275fc46(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2133, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c68dacd5dd48ba8d193b34417e96cec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3669e8652e2498122420bc5f8275fc46
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_563554d6805fb84fca60c4017cbb030b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2133, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c799b6daffbde8021f2c081d33fcffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_563554d6805fb84fca60c4017cbb030b
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c68dacd5dd48ba8d193b34417e96cec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3669e8652e2498122420bc5f8275fc46
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4031cb24998975a37a62b450724b9aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd341c26c429be4deceb7b7802859f50
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14792421460151672, 0.060222066938877106, 0.3426845371723175, 0.2729192078113556], [0.14792421460151672, 0.060222066938877106, 0.3426845371723175, 0.2729192078113556], [0.20768477022647858, 0.10328125953674316, 0.14738108217716217, 0.3912164270877838], [0.4820886254310608, 0.09006587415933609, 0.2595599591732025, 0.43800047039985657], [0.45774030685424805, 0.1348615139722824, 0.3947620391845703, 0.23843154311180115], [0.16375482082366943, 0.475871205329895, 0.369335412979126, 0.20670752227306366], [0.354208767414093, 0.34655365347862244, 0.2486346811056137, 0.16356748342514038]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.11575445532798767, 0.016847269609570503, 0.3592566251754761, 0.3711000978946686], [0.11575445532798767, 0.016847269609570503, 0.3592566251754761, 0.3711000978946686], [0.05424733832478523, 0.4543360769748688, 0.3827211558818817, 0.4861321449279785], [0.3619614541530609, 0.17202797532081604, 0.3347017168998718, 0.004203255288302898], [0.39603060483932495, 0.3538707494735718, 0.27019816637039185, 0.40859922766685486], [0.24719764292240143, 0.20523923635482788, 0.3153506815433502, 0.08138352632522583], [0.4453897476196289, 0.10803820192813873, 0.32960209250450134, 0.17188863456249237]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_d4f295f2f90b03961efed4f805eccdc2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[4631, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c30cb71eb1b7b2038a83fb2917733bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f295f2f90b03961efed4f805eccdc2
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f273faae99388a9b5dcbefa5964864c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4631, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4e7ead553d5a33d5fc604844c52f057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f273faae99388a9b5dcbefa5964864c7
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4c30cb71eb1b7b2038a83fb2917733bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f295f2f90b03961efed4f805eccdc2
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df3d523e1bcbca00cdaa412bb2831ca4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1039, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92cd16cb24904f329b2ccc556ec916bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df3d523e1bcbca00cdaa412bb2831ca4
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1039, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3e87178a498e5e0249a86aff5175d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d655b8c73ea81cb29893cee3aa5f9533
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_92cd16cb24904f329b2ccc556ec916bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df3d523e1bcbca00cdaa412bb2831ca4
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7617c4e25a0a20d178d0f697a9f7c6a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e17b6d4406f6436b786e9839fcef53
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.30025285482406616, 0.27773964405059814, 0.12615980207920074, 0.10717370361089706], [0.35898715257644653, 0.3946774899959564, 0.3250655233860016, 0.29423171281814575], [0.35898715257644653, 0.3946774899959564, 0.3250655233860016, 0.29423171281814575], [0.0367368683218956, 0.11908857524394989, 0.15985457599163055, 0.2605140507221222], [0.30330580472946167, 0.4362056255340576, 0.18896079063415527, 0.44536158442497253], [0.2442024052143097, 0.21146763861179352, 0.25890666246414185, 0.26765894889831543]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.4743831753730774, 0.2996712923049927, 0.05848970636725426, 0.21108603477478027], [0.009142567403614521, 0.35248109698295593, 0.30357590317726135, 0.11087937653064728], [0.009142567403614521, 0.35248109698295593, 0.30357590317726135, 0.11087937653064728], [0.11678148061037064, 0.35593029856681824, 0.24865007400512695, 0.439622163772583], [0.2514708936214447, 0.31273120641708374, 0.049657903611660004, 0.19320592284202576], [0.45483604073524475, 0.08000791817903519, 0.41738295555114746, 0.24432004988193512]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_0de2c0f203074d343e5e8d2822cd37bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5195df1055a103146ed731f087a9e76
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.7734333872795105, 2.265752077102661, 0.539688229560852, 6.51957893371582], [2.907357931137085, 0.14618602395057678, 0.6784908771514893, 2.3595669269561768]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_df64c8f14d2184bfcb25b966e58e2373(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de0324bed078bcbd34522cd4edfe7a49
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1.3565759658813477, 3.342909336090088, 1.0730841159820557, 0.8684794306755066], [4.103296756744385, 2.149132251739502, 0.5966194868087769, 0.18993321061134338]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_981a6ff83db0333fd3a29dc54d4f83ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.3625732958316803], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_945c5e2bb11b5b12aa1b8c23e3928b38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26832517981529236], [0.16028515994548798], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_59c57c1c340e6d846ee484a5edf8e6b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.33523043990135193], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_a8eb7d86a71ee343005edc45f423bc48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.16028515994548798], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.40533360838890076], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_cae1c7103b49c44b1a011c2eb5f4fc1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.3625732958316803], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e11d355f3cad765e47dcd5ab29656621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26832517981529236], [0.4581025242805481], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_348d1c167be938397108326ac99973d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06795760244131088], [0.04772914946079254], [-0.06907474249601364], [-0.05889728665351868], [0.017041902989149094]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_aed7428b19d9e44686556d073de8f5a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.33523043990135193], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_ec6efb252a3c4d429e01e0e7a6621397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.4581025242805481], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.40533360838890076], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_39b3cd44ae7a76ac0e8cf8bff3836342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.03523365408182144], [0.05821434408426285], [0.019736213609576225], [-0.04269413650035858], [0.0002444145502522588]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.06795760244131088], [0.04772914946079254], [-0.06907474249601364], [-0.05889728665351868], [0.017041902989149094]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b17bcaf8befdbcd243b4316594cb4c8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_792a19c7607a4198d6abfd687ef64154
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.9287696480751038], [0.18011359870433807], [4.499898433685303], [-0.37951698899269104], [-68.72540283203125]], dtype='float32').reshape([5, 1]),
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


    
    class PrimitiveOp_09ff84cafacfefa1df567c62e0acd188(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2318, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c12ba3acfa3b9752466a81b17a96ee8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09ff84cafacfefa1df567c62e0acd188
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2318, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5325e6be4edb3b02f5a390aa19b161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2ebb3b9756c653332ab77aad7c07a79
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c12ba3acfa3b9752466a81b17a96ee8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09ff84cafacfefa1df567c62e0acd188
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c52d50a1a8759388581e94cc6a87fad9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2961, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e672519cb281abd195b1c4a4c16689ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c52d50a1a8759388581e94cc6a87fad9
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2961, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16251d97299634a4e1d1d41a70863e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d83a11cdddb2822b0cd9543bdedf00
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e672519cb281abd195b1c4a4c16689ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c52d50a1a8759388581e94cc6a87fad9
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_33b1f0e724876d04da8e22a875c11e8e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[3739, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_304988b0850cfb65562f9c910b280761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33b1f0e724876d04da8e22a875c11e8e
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3739, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047df3010b09342a5279a38a0d68f365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fc2efbc3b73a8d0752b7f60726722f1
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_304988b0850cfb65562f9c910b280761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33b1f0e724876d04da8e22a875c11e8e
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_bcb28569648c8c5b1977a3b92a4a339a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ee6c518cc2c17722e81ecf9fb65043
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.26205939054489136, 0.14697584509849548, 0.4793679714202881, 0.02872663363814354, 0.21173322200775146, 0.23947587609291077, 0.37367042899131775, 0.32054823637008667, 0.2866531014442444, 0.07623961567878723, 0.13533982634544373, 0.0694308876991272, 0.11348035931587219, 0.21457159519195557, 0.39497604966163635, 0.06678082793951035, 0.23896890878677368, 0.24635815620422363, 0.29478490352630615, 0.0675901472568512], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_4c24b9b8bed3fcff1dfb206bea1cfeb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6ee6c518cc2c17722e81ecf9fb65043
        def get_inputs(self):
            return [
                paddle.to_tensor([0.26205939054489136, 0.14697584509849548, 0.4793679714202881, 0.02872663363814354, 0.21173322200775146, 0.23947587609291077, 0.37367042899131775, 0.32054823637008667, 0.2866531014442444, 0.07623961567878723, 0.13533982634544373, 0.0694308876991272, 0.11348035931587219, 0.21457159519195557, 0.39497604966163635, 0.06678082793951035, 0.23896890878677368, 0.24635815620422363, 0.29478490352630615, 0.0675901472568512], dtype='float32').reshape([20]),
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


    class TestPrimitiveOp_0d500ab78ebd4625732d81b2591b9fee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09640353918075562], [0.021595124155282974], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41484373807907104], [0.09736814349889755], [0.3636853098869324], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9c432ed2977b0e7004570cb7ed8917a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.061657778918743134], [0.08861131221055984], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_cdc941c8d0afa590883f830611b54ca0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.021595124155282974], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41484373807907104], [0.09475763142108917], [0.012042115442454815], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_ad952f23c9f102f260285087aef633a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.3512554168701172], [0.08861131221055984], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_eed9403e9951d9f3c6c28510887ec4a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09640353918075562], [0.25064805150032043], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09736814349889755], [0.3636853098869324], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_c88d00d64272b40ca0fec9c25248e5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22977420687675476], [0.061657778918743134], [0.2658151686191559], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_e0645946e7d6479f043a1aa054d5d21a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09553957730531693], [-0.059163011610507965], [-0.05419258028268814], [0.01992667280137539]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_b5ec128b8479c7e4c03f8f1036f26e19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.25064805150032043], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09475763142108917], [0.012042115442454815], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a7cc2b8428a98b396b6d76ad6038668f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22977420687675476], [0.3512554168701172], [0.2658151686191559], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9ff4d50ca749a48622f64438f479042f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.012408342212438583], [0.04858570545911789], [-0.009754105471074581], [0.04100790247321129]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09553957730531693], [-0.059163011610507965], [-0.05419258028268814], [0.01992667280137539]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d8a798ec2b97a91b64f26c2d112dd6ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b6d3e6784913d92fda481865d163c9
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-6.699625015258789], [2.2177040576934814], [-4.555873870849609], [0.5140772461891174]], dtype='float32').reshape([4, 1]),
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


    
    class PrimitiveOp_6224f15a5b4346e743eb9cdd2e094277(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[2013, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d86aede66419b531c0bc7ca281ee092d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6224f15a5b4346e743eb9cdd2e094277
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_56aa4f70b104214b9b3419834e480697(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[2013, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e1d50590f2a6e191bdd1df99b04d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56aa4f70b104214b9b3419834e480697
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d86aede66419b531c0bc7ca281ee092d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6224f15a5b4346e743eb9cdd2e094277
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7c4cd94f65c5efbb80d4b02eb5bd06a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3ba26ff94135edb3b0814ad6f72bc5
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40423333644866943, 0.28136950731277466, 0.12250851094722748, 0.0002954275696538389], [0.11568618565797806, 0.27895891666412354, 0.07727570831775665, 0.30709120631217957], [0.033719103783369064, 0.21601463854312897, 0.21620410680770874, 0.25814059376716614], [0.033719103783369064, 0.21601463854312897, 0.21620410680770874, 0.25814059376716614], [0.3263203501701355, 0.06131473928689957, 0.4514027237892151, 0.009571080096065998]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.03664267063140869, 0.14573366940021515, 0.043165650218725204, 0.07306309789419174], [0.4919965863227844, 0.31887757778167725, 0.07879022508859634, 0.3835740387439728], [0.11035966128110886, 0.15233902633190155, 0.18884484469890594, 0.39235854148864746], [0.11035966128110886, 0.15233902633190155, 0.18884484469890594, 0.39235854148864746], [0.06388995051383972, 0.09963720291852951, 0.2488420158624649, 0.4360834062099457]], dtype='float32').reshape([5, 4]),
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


    
    class PrimitiveOp_ed298ac667fe54d0f488487c2be2010e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[4177, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0dbb6c332e33fe6837c6a480c484db4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed298ac667fe54d0f488487c2be2010e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[4177, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac560c03ec11ea384051657165604b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5f3f260e171b0f595505b1c4ea23ec2
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0dbb6c332e33fe6837c6a480c484db4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed298ac667fe54d0f488487c2be2010e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_959e6cafcfd0128d6138b5fc4b61123b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd341c26c429be4deceb7b7802859f50
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16710375249385834, 0.3140609860420227, 0.03993195667862892, 0.4403045177459717], [0.18602943420410156, 0.47431832551956177, 0.39921894669532776, 0.28117549419403076], [0.026671169325709343, 0.3489063084125519, 0.31338927149772644, 0.1287972778081894], [0.16710375249385834, 0.3140609860420227, 0.03993195667862892, 0.4403045177459717], [0.45457276701927185, 0.10955125093460083, 0.07635635882616043, 0.00366982095874846], [0.08987396210432053, 0.48809531331062317, 0.3292236626148224, 0.40736350417137146], [0.45457276701927185, 0.10955125093460083, 0.07635635882616043, 0.00366982095874846]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.3205306828022003, 0.15136130154132843, 0.20435881614685059, 0.08078902214765549], [0.27505624294281006, 0.47417762875556946, 0.1448349505662918, 0.062408119440078735], [0.31028637290000916, 0.10561846196651459, 0.38038167357444763, 0.11800261586904526], [0.3205306828022003, 0.15136130154132843, 0.20435881614685059, 0.08078902214765549], [0.11132610589265823, 0.044622235000133514, 0.33257749676704407, 0.20738846063613892], [0.48840150237083435, 0.40047183632850647, 0.256814569234848, 0.2609950304031372], [0.11132610589265823, 0.044622235000133514, 0.33257749676704407, 0.20738846063613892]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_1edf3532ee2c65867b2ab6dbbb775a2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.1631958782672882]], [[0.4830591678619385]], [[0.13640281558036804]], [[0.06289021670818329]], [[0.3381606638431549]], [[0.1730637103319168]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6873317360877991]], [[0.5819525718688965]], [[0.7999815344810486]], [[0.5466142296791077]], [[0.614906907081604]], [[0.6087080240249634]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_9c33ec4410c114c7c46c7c4d41125c58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.16262555122375488]], [[0.4710995554924011]], [[0.46580007672309875]], [[0.40595364570617676]], [[0.4132225513458252]], [[0.436888724565506]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6331064105033875]], [[0.6973821520805359]], [[0.7257800102233887]], [[0.5502040386199951]], [[0.7628018856048584]], [[0.7564184069633484]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_ae3c7edd9669d3e65a2deb56f1a0d5ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.44883012771606445, 0.1614665985107422]], [[0.4670383036136627, 0.20265744626522064]], [[0.20590291917324066, 0.1419561803340912]], [[0.24326932430267334, 0.28298479318618774]], [[0.3838028907775879, 0.49517402052879333]], [[0.48496755957603455, 0.13528983294963837]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.3943357467651367, 0.0671614482998848]], [[0.47695010900497437, 0.19709175825119019]], [[0.0027434383518993855, 0.3336830139160156]], [[0.2079385668039322, 0.07217258960008621]], [[0.1960109919309616, 0.05200967937707901]], [[0.20042534172534943, 0.4425562620162964]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_faa1d929f4ad8bde648d254fadf85033(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.08073487877845764, 0.32428503036499023]], [[0.3015163540840149, 0.4314993619918823]], [[0.16413716971874237, 0.15354189276695251]], [[0.07749584317207336, 0.1363573968410492]], [[0.16635501384735107, 0.1668197214603424]], [[0.08065397292375565, 0.22361506521701813]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.3943357467651367, 0.0671614482998848]], [[0.47695010900497437, 0.19709175825119019]], [[0.0027434383518993855, 0.3336830139160156]], [[0.2079385668039322, 0.07217258960008621]], [[0.1960109919309616, 0.05200967937707901]], [[0.20042534172534943, 0.4425562620162964]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_a96d85f42cbd03250c4b9582fb562421(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.16726118326187134, 0.34479621052742004]], [[0.3814340829849243, 0.18334932625293732]], [[0.3706839680671692, 0.2882494330406189]], [[0.20342600345611572, 0.3141588270664215]], [[0.40584495663642883, 0.05891812965273857]], [[0.4210466742515564, 0.15735946595668793]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_b2ff266e9919e71a82daf03a57bfd65c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.05183764174580574, 0.1322854906320572, 0.3363713026046753, 0.13401319086551666, 0.47789466381073, 0.2742125988006592, 0.16450469195842743, 0.27145132422447205, 0.2347048819065094, 0.46265271306037903, 0.4377806782722473, 0.1997579038143158, 0.23424941301345825, 0.18413367867469788, 0.47968658804893494, 0.4680825173854828], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_76e3726c975db230f3b6fa73a4aeb8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.05183764174580574, 0.1322854906320572, 0.3363713026046753, 0.13401319086551666, 0.47789466381073, 0.2742125988006592, 0.16450469195842743, 0.27145132422447205, 0.2347048819065094, 0.46265271306037903, 0.4377806782722473, 0.1997579038143158, 0.23424941301345825, 0.18413367867469788, 0.47968658804893494, 0.4680825173854828], dtype='float32').reshape([16]),
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


    class TestPrimitiveOp_b56965715626a0da39f4afe5332367ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a8746c3beac4d26440f15471ec9fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b56965715626a0da39f4afe5332367ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01fbc341750f8c685ca87b800f5f64f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03942854702472687, 0.47896647453308105, 0.0707368552684784, 0.15708090364933014], [0.34924668073654175, 0.4959641695022583, 0.2033059000968933, 0.2109263688325882], [0.31559088826179504, 0.3385964035987854, 0.3923597037792206, 0.36877623200416565], [0.33733460307121277, 0.14558610320091248, 0.30153128504753113, 0.06012469157576561], [0.35864126682281494, 0.227564737200737, 0.30506813526153564, 0.17007820308208466]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.09926425665616989, 0.23578712344169617, 0.13742288947105408, 0.06377401947975159], [0.37992334365844727, 0.08063960075378418, 0.30269384384155273, 0.11003783345222473], [0.35751020908355713, 0.22279569506645203, 0.25694164633750916, 0.4777662456035614], [0.13734689354896545, 0.49705782532691956, 0.2898792028427124, 0.38485613465309143], [0.254555881023407, 0.04162680357694626, 0.20142796635627747, 0.09924576431512833]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_2352dfd20599fa40540a807fa1dfb7fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28719210624694824, 0.23055420815944672, 0.04956628754734993, 0.22986920177936554], [0.29731088876724243, 0.4535277485847473, 0.4582275152206421, 0.4929276704788208], [0.4882139265537262, 0.21820244193077087, 0.09530343115329742, 0.26205146312713623], [0.29731088876724243, 0.4535277485847473, 0.4582275152206421, 0.4929276704788208], [0.4882139265537262, 0.21820244193077087, 0.09530343115329742, 0.26205146312713623]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.3666442632675171, 0.038568176329135895, 0.14608372747898102, 0.28126558661460876], [0.43499240279197693, 0.4896965026855469, 0.4885249137878418, 0.4071301221847534], [0.42328980565071106, 0.2418367564678192, 0.06388422101736069, 0.4153708517551422], [0.43499240279197693, 0.4896965026855469, 0.4885249137878418, 0.4071301221847534], [0.42328980565071106, 0.2418367564678192, 0.06388422101736069, 0.4153708517551422]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_a05ff82779dd26852f80166db83b3eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.046371277421712875], [0.0077659436501562595], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.047508757561445236], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.18618053197860718], [0.042280279099941254], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_b6c13f95ebfa7cff8df687a2cb70894d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42142167687416077], [0.338043212890625], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.2514757812023163], [0.4964841306209564], [0.21071745455265045], [0.1466875672340393], [0.4495491087436676], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_4351eea4ee042ad2a2a3c6a344c4b021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36666956543922424], [0.36908185482025146], [0.2961925268173218], [0.035441603511571884], [0.4349591135978699], [0.2363598495721817], [0.18706205487251282], [0.06219051405787468], [0.03899889066815376]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.18618053197860718], [0.007991598919034004], [0.08734893053770065], [0.48177430033683777], [0.4414879083633423], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4638229012489319]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_d19d9aa3fa954ae2d05d91586a20097e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.338043212890625], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.3057922422885895], [0.23234502971172333], [0.08419803529977798]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.2514757812023163], [0.4430234134197235], [0.21071745455265045], [0.002510953461751342], [0.4495491087436676], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_405078fc70779c93cef045fc62ee2003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4386983811855316], [0.046371277421712875], [0.0077659436501562595], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.047508757561445236], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.042280279099941254], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.3348342180252075], [0.45164355635643005], [0.08683238923549652], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_57e7727d5c393d3e0fea3d8ef867acbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42142167687416077], [0.39432990550994873], [0.012163503095507622], [0.2716471254825592], [0.08528527617454529], [0.09083577990531921], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3526066541671753], [0.4370092451572418], [0.11843360215425491], [0.4964841306209564], [0.09898043423891068], [0.1466875672340393], [0.21377909183502197], [0.2298479974269867], [0.3915681838989258]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_e910b9418427bd8fc6696b702110f831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06809649616479874], [0.1168467253446579], [0.05836332216858864], [0.04395139962434769], [-0.006267378106713295], [0.004919853061437607], [-0.005880832672119141], [-0.006536051165312529], [0.1567060500383377]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.012420357204973698], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_ca1fd952653f8ac34098bb032c9f47b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4386983811855316], [0.36908185482025146], [0.2961925268173218], [0.29862847924232483], [0.4404768645763397], [0.39816227555274963], [0.3323768377304077], [0.06219051405787468], [0.04453969746828079]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0328356958925724], [0.007991598919034004], [0.08102629333734512], [0.3775736689567566], [0.0860084742307663], [0.1868826150894165], [0.3090613782405853], [0.07356120645999908], [0.4006982743740082]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_08d48b712c50a1eefd14e30f5fd6d524(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4806118905544281], [0.39432990550994873], [0.493656724691391], [0.3843192160129547], [0.4271245300769806], [0.173434779047966], [0.41013792157173157], [0.3947860300540924], [0.2945002317428589]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.25806623697280884], [0.013965434394776821], [0.11843360215425491], [0.4430234134197235], [0.09898043423891068], [0.002510953461751342], [0.21377909183502197], [0.22793996334075928], [0.3716922998428345]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9e498f62dda68437b25faae6a928a937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09032297879457474], [0.13734589517116547], [0.08073534816503525], [0.004634413868188858], [0.11631671339273453], [0.03611272946000099], [0.004578196443617344], [-0.0018971551908180118], [0.027492618188261986]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.055676139891147614], [0.1168467253446579], [0.05836332216858864], [0.04395139962434769], [-0.006267378106713295], [0.004919853061437607], [-0.005880832672119141], [-0.006536051165312529], [0.1567060500383377]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_93fac07e4e4a18e6c488efb731c340cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22308222949504852], [0.0], [0.0], [0.0], [-0.0], [0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.3835883140563965], [0.14925214648246765], [0.2771032452583313], [-8.483701705932617], [1.0538820028305054], [0.8637640476226807], [2.2845304012298584], [-2.4451851844787598], [-4.699932098388672]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_1af58d553c5e7b6ec645021f13cf8a4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_255402753063472a8b72eceb8247eefe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.24009037017822266]], [[0.36598658561706543]], [[0.45717522501945496]], [[0.01734936609864235]], [[0.4299929738044739]], [[0.46376073360443115]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6013274788856506]], [[0.5645684599876404]], [[0.6824908256530762]], [[0.7009567022323608]], [[0.7100949287414551]], [[0.6275843977928162]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_9989a99a60665a5fe03f40208d83bb1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4855091869831085]], [[0.48198366165161133]], [[0.10854921489953995]], [[0.05358770489692688]], [[0.39330196380615234]], [[0.21844549477100372]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6730449199676514]], [[0.807695209980011]], [[0.7492591738700867]], [[0.6105006337165833]], [[0.7242305874824524]], [[0.5755680203437805]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_3938bc10f243ced1b56a73317bb0e0ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70dee0cd33c5def9be5706494a26831d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ab9198f64366df883d7c22161b259a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_70dee0cd33c5def9be5706494a26831d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbf848776a40e812e4866203f4f9150f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22895076870918274, 0.43839433789253235, 0.09775096923112869, 0.4802975058555603], [0.3041378855705261, 0.25429245829582214, 0.43246856331825256, 0.3057610094547272], [0.4121575355529785, 0.4209367334842682, 0.025442074984312057, 0.0592651404440403], [0.3041378855705261, 0.25429245829582214, 0.43246856331825256, 0.3057610094547272], [0.4121575355529785, 0.4209367334842682, 0.025442074984312057, 0.0592651404440403], [0.286880224943161, 0.007150974590331316, 0.1812123954296112, 0.08457574993371964], [0.286880224943161, 0.007150974590331316, 0.1812123954296112, 0.08457574993371964]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.12005609273910522, 0.18032310903072357, 0.24188463389873505, 0.0608525313436985], [0.003236803924664855, 0.434544175863266, 0.31264448165893555, 0.36690962314605713], [0.04446536302566528, 0.2904546856880188, 0.4626550078392029, 0.05316312983632088], [0.003236803924664855, 0.434544175863266, 0.31264448165893555, 0.36690962314605713], [0.04446536302566528, 0.2904546856880188, 0.4626550078392029, 0.05316312983632088], [0.1808030754327774, 0.28809061646461487, 0.29978424310684204, 0.401498407125473], [0.1808030754327774, 0.28809061646461487, 0.29978424310684204, 0.401498407125473]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_a6b307b98a81c3cb5e2d3782c9c0a038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.04505692422389984, 0.4173981845378876, 0.0506727397441864, 0.004974461626261473, 0.0021464754827320576], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e7370d269344c78293369d886645c1aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.27147242426872253, 0.34147074818611145, 0.1349279284477234, 0.04442029446363449, 0.05202075466513634, 0.2485688030719757], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b0dde865db2b12ffda7eb548d45e24a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.3307744562625885, 0.4149071276187897, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c49ab63c0f37c7c1c395ff0f954a60d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.4658578038215637, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4fb159eb90dbe4315554fe059c8cf65b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.19085991382598877, 0.13665537536144257, 0.2634378671646118, 0.18630926311016083], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1420275717973709, 0.3307744562625885, 0.4149071276187897, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4ecc6434e46daa88692ec15cd9bcf5a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23692236840724945, 0.3351680040359497, 0.10397005081176758, 0.09553638845682144, 0.20359452068805695, 0.42669540643692017], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.4658578038215637, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_aebd32b73ee0afab73056eed93688b63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32582366466522217, 0.13624942302703857, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1420275717973709, 0.13624942302703857, 0.2611750066280365, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_baadb69fa6bb53d57921a039bd2fecaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_61f9a57961b58eb0fd23b597f68cf79c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.04189087823033333, 0.03793960437178612, -0.019696950912475586, -0.00199795956723392, -0.005094417370855808, -0.05107930675148964], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e61deb9c76ef61c92651abddcbccffbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.23392561078071594, 0.09065317362546921, 0.33928659558296204, 0.11514031887054443, 0.14891447126865387, 0.22644412517547607], dtype='float32').reshape([6]),
                paddle.to_tensor([0.19883015751838684, 0.4130263328552246, 0.302883505821228, 0.13395777344703674, 0.13623738288879395, 0.2516404986381531], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_e57ba4488d9926ef9ac3d6b03bde82cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3495340347290039, 0.3608149290084839, 0.2594084143638611, 0.25006911158561707, 0.2193455547094345, 0.35870659351348877], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3054583668708801, 0.21985271573066711, 0.06001290678977966, 0.2806971073150635, 0.21360710263252258, 0.2312326282262802], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_2b44e8c23602aa897759c3b74822477f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.35163626074790955, 0.4952782392501831, 0.4173981845378876, 0.17960789799690247, 0.2928544878959656, 0.4507417678833008], dtype='float32').reshape([6]),
                paddle.to_tensor([0.04602406919002533, 0.13624942302703857, 0.2611750066280365, 0.13126017153263092, 0.009036889299750328, 0.3169717490673065], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a95bb0e6476e72a1c45be4c304f3763d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4275956451892853, 0.38015908002853394, 0.3838888704776764, 0.45571792125701904, 0.3866703510284424, 0.4688444137573242], dtype='float32').reshape([6]),
                paddle.to_tensor([0.373994380235672, 0.10453741252422333, 0.016055766493082047, 0.45571792125701904, 0.22361966967582703, 0.03576983883976936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_3d382a66129aed15889077206014316e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.149177074432373, 0.6195820569992065, -1.1968660354614258, -0.014567945152521133, -1.4922434091567993, -0.3225652575492859], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.8666291832923889, 1.1695619821548462, -0.5603955984115601, 0.3037809133529663, 0.7104108333587646, 1.1143471002578735], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0399b2f0df1902d591405b73d4652512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b81bb79374245a2c3ad17c509e4e02b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0399b2f0df1902d591405b73d4652512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3738802e3b2affa6656be927be90b8b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1bf525165d16c2a153a298dc0916bf81
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec72a5adb5bb15bb7b0443e0b5bd5772(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.06644057482481003, 0.0018614486325532198, 0.21422740817070007, 0.006235000677406788, 0.3750665485858917, 0.19984327256679535, 0.4298643171787262, 0.3690134882926941, 0.419566810131073, 0.12179119884967804, 0.4229738712310791, 0.4515219032764435, 0.41391289234161377, 0.24803227186203003, 0.41406798362731934, 0.24153155088424683, 0.07568644732236862, 0.3699430823326111, 0.23401470482349396, 0.25811436772346497, 0.40426042675971985, 0.43211135268211365, 0.1867329478263855, 0.0654502809047699], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_b9456f11f6dcdf1b7f424725ab1046e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.06644057482481003, 0.0018614486325532198, 0.21422740817070007, 0.006235000677406788, 0.3750665485858917, 0.19984327256679535, 0.4298643171787262, 0.3690134882926941, 0.419566810131073, 0.12179119884967804, 0.4229738712310791, 0.4515219032764435, 0.41391289234161377, 0.24803227186203003, 0.41406798362731934, 0.24153155088424683, 0.07568644732236862, 0.3699430823326111, 0.23401470482349396, 0.25811436772346497, 0.40426042675971985, 0.43211135268211365, 0.1867329478263855, 0.0654502809047699], dtype='float32').reshape([24]),
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


    class TestPrimitiveOp_b479eda1c41306328b7b1add88411977(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8272a7c522dee0eba2e941986046765b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b479eda1c41306328b7b1add88411977(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_995dfbca5b61b5719434a1503e104f85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.3004399240016937, 0.37846723198890686, 0.201074481010437, 0.014919416047632694], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_01119198d954be80f32e19b2c7b28ca0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3004399240016937, 0.37846723198890686, 0.201074481010437, 0.014919416047632694], dtype='float32').reshape([4]),
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


    class TestPrimitiveOp_64170e6c21b3a4548689c466f3dc8d6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4689941704273224, 0.284049928188324, 0.1052517294883728, 0.053699977695941925], [0.3941596746444702, 0.1582547277212143, 0.39884933829307556, 0.03849705308675766], [0.13040339946746826, 0.043503545224666595, 0.043750714510679245, 0.13780781626701355], [0.33933836221694946, 0.1820559799671173, 0.24868448078632355, 0.006877193693071604], [0.33933836221694946, 0.1820559799671173, 0.24868448078632355, 0.006877193693071604], [0.13040339946746826, 0.043503545224666595, 0.043750714510679245, 0.13780781626701355]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.2394210398197174, 0.4624246060848236, 0.0883968323469162, 0.46321913599967957], [0.36362117528915405, 0.18431217968463898, 0.3945719599723816, 0.13577550649642944], [0.3852720856666565, 0.3568287193775177, 0.2815832197666168, 0.3905746340751648], [0.185911163687706, 0.23607124388217926, 0.44473662972450256, 0.088658407330513], [0.185911163687706, 0.23607124388217926, 0.44473662972450256, 0.088658407330513], [0.3852720856666565, 0.3568287193775177, 0.2815832197666168, 0.3905746340751648]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_ad1d365b1d764be9549fa58b020d9883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3586193919181824, 0.13367393612861633, 0.07783482223749161, 0.09604031592607498], [0.2363462895154953, 0.015636775642633438, 0.05108630657196045, 0.4619491994380951], [0.19445933401584625, 0.4037427604198456, 0.3171370029449463, 0.12906880676746368], [0.465982049703598, 0.4749557673931122, 0.1601065844297409, 0.08161168545484543], [0.3586193919181824, 0.13367393612861633, 0.07783482223749161, 0.09604031592607498]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.14423157274723053, 0.059380531311035156, 0.3358438014984131, 0.3124944567680359], [0.3744156062602997, 0.44170665740966797, 0.051465265452861786, 0.0004932225565426052], [0.07123012840747833, 0.2832140326499939, 0.02902403473854065, 0.33228036761283875], [0.4094831347465515, 0.4135250747203827, 0.021371887996792793, 0.3999515473842621], [0.14423157274723053, 0.059380531311035156, 0.3358438014984131, 0.3124944567680359]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_73ece6d168a08bd3fa887a7c95aa72af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5ccf27dca01f6f43bde49c34f209a36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b9d10b48039d174609285673c7470fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_ec593f2bce8cc31cc74bc1bbe4979185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.4937298595905304]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2bc82ddbac90a0198fe03105a23d1401(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_46890465cca594d33c601a0282add8a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2893354296684265]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_b9d10b48039d174609285673c7470fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.23339727520942688]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.21023254096508026]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_f98357e61dd70738828421f9496e218b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.002530277008190751]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_00c126ef2f810ac5c2bb49e2fb7daaff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4828311502933502]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01672634482383728]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_2bc82ddbac90a0198fe03105a23d1401(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4335895776748657]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.08633378893136978]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_fda38fb673d7013a8621f54eb031b22e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16185759007930756]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.002530277008190751]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_04c438cf45bfb6c7c79b1c977637c941(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.9843672513961792]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_db506fcf3e195100965400c62da71968(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.13643445074558258], [0.41118323802948], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_5320bc9d7c44efd1c9ba63ef5a8b1854(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.09585190564393997], [0.27821820974349976], [0.4414989948272705], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.42264029383659363], [0.4681013226509094], [0.35276615619659424], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_801b28f8820e614555af8ebe7540acbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13170580565929413], [0.058484263718128204], [0.06756104528903961], [0.14288948476314545], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.13643445074558258], [0.08216515928506851], [0.41876524686813354], [0.46803197264671326], [0.20365557074546814], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8fab3dfe33cbed4d49dfe6d6819807be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.09585190564393997], [0.27821820974349976], [0.4517127275466919], [0.20359814167022705]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.42264029383659363], [0.03228634223341942], [0.35276615619659424], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_08bcd1acb1d6c0a796e8710c05fca718(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.2057957798242569], [0.31516075134277344]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.41118323802948], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.34649741649627686]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2dc3819f39454f74b30332cf8abd4a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.34790247678756714], [0.01755298487842083], [0.2810487151145935], [0.3574141561985016], [0.4414989948272705], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.35616353154182434], [0.3317866325378418], [0.39316296577453613], [0.4681013226509094], [0.0868077278137207], [0.19587846100330353]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_c1f889363b829a1de8b66326e814c024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0036731057334691286], [0.035216521471738815], [0.10041127353906631], [-0.07269255071878433], [0.07417459785938263], [-0.0034735659137368202]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.00018990683020092547], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_0921ff70fb6b3c790643c14cee96832e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20599016547203064], [0.29832562804222107], [0.4217703342437744], [0.34355056285858154], [0.4329535663127899], [0.38953283429145813]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0025779781863093376], [0.08216515928506851], [0.2937028706073761], [0.40923434495925903], [0.060637570917606354], [0.2550865709781647]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_2b195a8d6e114fb847b209b91929c613(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.48313620686531067], [0.30361756682395935], [0.2810487151145935], [0.3574141561985016], [0.4517127275466919], [0.43758612871170044]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.06172452121973038], [0.29318130016326904], [0.39316296577453613], [0.03228634223341942], [0.0868077278137207], [0.1730971336364746]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_11cd53271498123ded99d49a82559b9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08572027832269669], [0.0022559084463864565], [-0.014358188025653362], [-0.02135562337934971], [0.13585996627807617], [0.035559557378292084]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.0036731057334691286], [0.035216521471738815], [0.10041127353906631], [-0.07269255071878433], [0.07398469001054764], [-0.0034735659137368202]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_38adeac3e64b0d8fc2e155066dfa61fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.0], [0.0], [-0.0], [0.0025668395683169365], [-0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[1.0428498983383179], [-14.610793113708496], [7.993310928344727], [-2.40390682220459], [0.4554342031478882], [1.097683072090149]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8147e6600d4809d0a0b71cfb47f08a2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.41194984316825867, 0.1985234022140503, 0.08035333454608917, 0.2548809051513672], [0.389752596616745, 0.49257543683052063, 0.24452143907546997, 0.4366907775402069], [0.4164615869522095, 0.11515739560127258, 0.4882013499736786, 0.2104092687368393], [0.04727732017636299, 0.46199488639831543, 0.14487098157405853, 0.17658351361751556]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.3674372732639313, 0.39325854182243347, 0.44171425700187683, 0.23020479083061218], [0.43334540724754333, 0.047591667622327805, 0.1316508948802948, 0.4297671318054199], [0.15011842548847198, 0.0014065280556678772, 0.15698941051959991, 0.06276851147413254], [0.20743593573570251, 0.3237133324146271, 0.13536261022090912, 0.3818800747394562]], dtype='float32').reshape([4, 4]),
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


    class TestPrimitiveOp_685d5e13ee1aee9f0a84dad81494b361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eaa175646e01635ad819d645b721187a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_685d5e13ee1aee9f0a84dad81494b361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67ee3dd02410dfeec83c2606c1d265ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14792421460151672, 0.060222066938877106, 0.3426845371723175, 0.2729192078113556], [0.14792421460151672, 0.060222066938877106, 0.3426845371723175, 0.2729192078113556], [0.20768477022647858, 0.10328125953674316, 0.14738108217716217, 0.3912164270877838], [0.4820886254310608, 0.09006587415933609, 0.2595599591732025, 0.43800047039985657], [0.45774030685424805, 0.1348615139722824, 0.3947620391845703, 0.23843154311180115], [0.16375482082366943, 0.475871205329895, 0.369335412979126, 0.20670752227306366], [0.354208767414093, 0.34655365347862244, 0.2486346811056137, 0.16356748342514038]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.11575445532798767, 0.016847269609570503, 0.3592566251754761, 0.3711000978946686], [0.11575445532798767, 0.016847269609570503, 0.3592566251754761, 0.3711000978946686], [0.05424733832478523, 0.4543360769748688, 0.3827211558818817, 0.4861321449279785], [0.3619614541530609, 0.17202797532081604, 0.3347017168998718, 0.004203255288302898], [0.39603060483932495, 0.3538707494735718, 0.27019816637039185, 0.40859922766685486], [0.24719764292240143, 0.20523923635482788, 0.3153506815433502, 0.08138352632522583], [0.4453897476196289, 0.10803820192813873, 0.32960209250450134, 0.17188863456249237]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_b034c4ab5d9a4f59546a6eca6d9fd80a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2239b319443d8f39f479b1c8e5a55887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b034c4ab5d9a4f59546a6eca6d9fd80a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c1fa9bdfb6f477795047cf90d5229e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d17f7fdd25536df690bcf7829c9e2bfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_5c1fa9bdfb6f477795047cf90d5229e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a89dfe01d48a19b39a14b931a6f22cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e421643d1e9a7221f991f14614920b8
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e8f1fce68a2ec0919891c25fad229f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.30025285482406616, 0.27773964405059814, 0.12615980207920074, 0.10717370361089706], [0.35898715257644653, 0.3946774899959564, 0.3250655233860016, 0.29423171281814575], [0.35898715257644653, 0.3946774899959564, 0.3250655233860016, 0.29423171281814575], [0.0367368683218956, 0.11908857524394989, 0.15985457599163055, 0.2605140507221222], [0.30330580472946167, 0.4362056255340576, 0.18896079063415527, 0.44536158442497253], [0.2442024052143097, 0.21146763861179352, 0.25890666246414185, 0.26765894889831543]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.4743831753730774, 0.2996712923049927, 0.05848970636725426, 0.21108603477478027], [0.009142567403614521, 0.35248109698295593, 0.30357590317726135, 0.11087937653064728], [0.009142567403614521, 0.35248109698295593, 0.30357590317726135, 0.11087937653064728], [0.11678148061037064, 0.35593029856681824, 0.24865007400512695, 0.439622163772583], [0.2514708936214447, 0.31273120641708374, 0.049657903611660004, 0.19320592284202576], [0.45483604073524475, 0.08000791817903519, 0.41738295555114746, 0.24432004988193512]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_7cb8fefd5e8a75b42de64a25a15f9a34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.7734333872795105, 2.265752077102661, 0.539688229560852, 6.51957893371582], [2.907357931137085, 0.14618602395057678, 0.6784908771514893, 2.3595669269561768]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_8fd6e881dd93519854e6b9aaa46cd8c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_423011ebd9e7ee90ee479c694ef3a796
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[1.3565759658813477, 3.342909336090088, 1.0730841159820557, 0.8684794306755066], [4.103296756744385, 2.149132251739502, 0.5966194868087769, 0.18993321061134338]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_bfca6e46c56e8ef23f6514ee0a4a778f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.3625732958316803], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f8eb4ee11549035ab8c380c6ad29b613(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26832517981529236], [0.16028515994548798], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_b4ab1f85ce1d6f1a2325916a9666b51b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29793399572372437], [0.45982950925827026], [0.041210729628801346], [0.08048146218061447], [0.11273333430290222]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.3187001049518585], [0.395175576210022], [0.4700484275817871], [0.33523043990135193], [0.4927985966205597]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_9c014d1c45920997863232e2eab02a78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.16028515994548798], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.40533360838890076], [0.32245928049087524], [0.2967213988304138], [0.3857642710208893], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e6b51e073612bfe31c9d4f40522c6e9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.3625732958316803], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_87583d0e13c76ec0db61c846310bb3a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.26832517981529236], [0.4581025242805481], [0.2240569293498993], [0.3378629684448242], [0.006186048965901136]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.49802151322364807], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.21820658445358276]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_bafc7ccac5ffada2f69f8a80cf0cdf9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06795760244131088], [0.04772914946079254], [-0.06907474249601364], [-0.05889728665351868], [0.017041902989149094]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e16a601d4362d92fc7e6825d329d74cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.44977208971977234], [0.4644966721534729], [0.2582728862762451], [0.2239593118429184], [0.40969327092170715]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1435127556324005], [0.09152258932590485], [0.2088557481765747], [0.33523043990135193], [0.48419061303138733]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_cc2bed9235e217efb3e6b5908f487b10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29028844833374023], [0.4581025242805481], [0.474911093711853], [0.49212008714675903], [0.08536812663078308]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.40533360838890076], [0.3020210564136505], [0.07553115487098694], [0.10842543095350266], [0.0886489748954773]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_12dbf1b47e85ba1c9d96839c9e1fb377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.03523365408182144], [0.05821434408426285], [0.019736213609576225], [-0.04269413650035858], [0.0002444145502522588]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.06795760244131088], [0.04772914946079254], [-0.06907474249601364], [-0.05889728665351868], [0.017041902989149094]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_298ac4f8b2bff361d07e93651e119a67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.9287696480751038], [0.18011359870433807], [4.499898433685303], [-0.37951698899269104], [-68.72540283203125]], dtype='float32').reshape([5, 1]),
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


    class TestPrimitiveOp_6d1ef172865f0b3ff5a2533d6eeda5e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db4d73c5f80e5341fbb3932a229c4cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_6d1ef172865f0b3ff5a2533d6eeda5e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6ec7679e7073c03b77259bbaddc7743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74757fada639e29dfb9514dbefc2cbf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b6ec7679e7073c03b77259bbaddc7743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d263e9fc622c2c18df877dd16f945cb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d717def0e8c6648b75a8eeb05ee3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d263e9fc622c2c18df877dd16f945cb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f74a83ba56eb3141df21b59e1afd71f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.26205939054489136, 0.14697584509849548, 0.4793679714202881, 0.02872663363814354, 0.21173322200775146, 0.23947587609291077, 0.37367042899131775, 0.32054823637008667, 0.2866531014442444, 0.07623961567878723, 0.13533982634544373, 0.0694308876991272, 0.11348035931587219, 0.21457159519195557, 0.39497604966163635, 0.06678082793951035, 0.23896890878677368, 0.24635815620422363, 0.29478490352630615, 0.0675901472568512], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_53675dea3dfd1a1a97e850b516cd6de6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e054784e9bd5a94ea12d4a3afff45fc1
        def get_inputs(self):
            return [
                paddle.to_tensor([0.26205939054489136, 0.14697584509849548, 0.4793679714202881, 0.02872663363814354, 0.21173322200775146, 0.23947587609291077, 0.37367042899131775, 0.32054823637008667, 0.2866531014442444, 0.07623961567878723, 0.13533982634544373, 0.0694308876991272, 0.11348035931587219, 0.21457159519195557, 0.39497604966163635, 0.06678082793951035, 0.23896890878677368, 0.24635815620422363, 0.29478490352630615, 0.0675901472568512], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_4708a4529cb2268842a29b94f60f078f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09640353918075562], [0.021595124155282974], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41484373807907104], [0.09736814349889755], [0.3636853098869324], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6da68b18e8b546d16b1c3af32656717f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.061657778918743134], [0.08861131221055984], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9b76c1a0aa7efaae790812370b6bda8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.021595124155282974], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.41484373807907104], [0.09475763142108917], [0.012042115442454815], [0.26613038778305054]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_0d269b7ee0bc6dccb608d42dc5fc5585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10462348163127899], [0.3512554168701172], [0.08861131221055984], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_89968f33d976645fc0ef466e8f763000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09640353918075562], [0.25064805150032043], [0.09111341089010239], [0.12773194909095764]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09736814349889755], [0.3636853098869324], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_bf4ab664171f3b354a940b5da1a46368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22977420687675476], [0.061657778918743134], [0.2658151686191559], [0.3008590340614319]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.441641241312027], [0.2988758683204651], [0.3646744191646576], [0.2739860415458679]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_042c3df5367c540ce9a8da57e6f2c2c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09553957730531693], [-0.059163011610507965], [-0.05419258028268814], [0.01992667280137539]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_403a032a88ab02c58b561da2046c0c1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2870025336742401], [0.25064805150032043], [0.4148816764354706], [0.3877279460430145]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3469667136669159], [0.09475763142108917], [0.012042115442454815], [0.14231333136558533]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_647fda7b8ecff27ca1d55bb7c59f4560(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22977420687675476], [0.3512554168701172], [0.2658151686191559], [0.4113050699234009]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.4367034435272217], [0.03958968445658684], [0.29002854228019714], [0.24420864880084991]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a8d976912321c5e2e5dc990f4e355965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.012408342212438583], [0.04858570545911789], [-0.009754105471074581], [0.04100790247321129]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.09553957730531693], [-0.059163011610507965], [-0.05419258028268814], [0.01992667280137539]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_195a934eec21889b667d7c7ad89dcb38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-6.699625015258789], [2.2177040576934814], [-4.555873870849609], [0.5140772461891174]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f866141f6138d4c2131ae7ec085de7de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69c7b94220fc52f97cc1ab860830f2fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9577250e3649ddb471ec67e68d825c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_69c7b94220fc52f97cc1ab860830f2fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_cfbcd12182d5b113a5a203da5c4cd018(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.40423333644866943, 0.28136950731277466, 0.12250851094722748, 0.0002954275696538389], [0.11568618565797806, 0.27895891666412354, 0.07727570831775665, 0.30709120631217957], [0.033719103783369064, 0.21601463854312897, 0.21620410680770874, 0.25814059376716614], [0.033719103783369064, 0.21601463854312897, 0.21620410680770874, 0.25814059376716614], [0.3263203501701355, 0.06131473928689957, 0.4514027237892151, 0.009571080096065998]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.03664267063140869, 0.14573366940021515, 0.043165650218725204, 0.07306309789419174], [0.4919965863227844, 0.31887757778167725, 0.07879022508859634, 0.3835740387439728], [0.11035966128110886, 0.15233902633190155, 0.18884484469890594, 0.39235854148864746], [0.11035966128110886, 0.15233902633190155, 0.18884484469890594, 0.39235854148864746], [0.06388995051383972, 0.09963720291852951, 0.2488420158624649, 0.4360834062099457]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_172cf9d131de26fd7fd1abde1ea90036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb0c55c71757d590c98b867815b833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_172cf9d131de26fd7fd1abde1ea90036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86525f90cb9182514e4dc4fdf6216bc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a22ed2bc5a616c3396fc86c456c1f7e3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16710375249385834, 0.3140609860420227, 0.03993195667862892, 0.4403045177459717], [0.18602943420410156, 0.47431832551956177, 0.39921894669532776, 0.28117549419403076], [0.026671169325709343, 0.3489063084125519, 0.31338927149772644, 0.1287972778081894], [0.16710375249385834, 0.3140609860420227, 0.03993195667862892, 0.4403045177459717], [0.45457276701927185, 0.10955125093460083, 0.07635635882616043, 0.00366982095874846], [0.08987396210432053, 0.48809531331062317, 0.3292236626148224, 0.40736350417137146], [0.45457276701927185, 0.10955125093460083, 0.07635635882616043, 0.00366982095874846]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.3205306828022003, 0.15136130154132843, 0.20435881614685059, 0.08078902214765549], [0.27505624294281006, 0.47417762875556946, 0.1448349505662918, 0.062408119440078735], [0.31028637290000916, 0.10561846196651459, 0.38038167357444763, 0.11800261586904526], [0.3205306828022003, 0.15136130154132843, 0.20435881614685059, 0.08078902214765549], [0.11132610589265823, 0.044622235000133514, 0.33257749676704407, 0.20738846063613892], [0.48840150237083435, 0.40047183632850647, 0.256814569234848, 0.2609950304031372], [0.11132610589265823, 0.044622235000133514, 0.33257749676704407, 0.20738846063613892]], dtype='float32').reshape([7, 4]),
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