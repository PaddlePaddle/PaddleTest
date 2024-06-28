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
    class PrimitiveOp_27da3eecadc783d5eca3f7e8577dea78(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd102e990d939cc5aab66ab65798c8e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27da3eecadc783d5eca3f7e8577dea78
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
                paddle.uniform([91], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa8f6222620e848633e07558c8fd7101(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7885c31a20c8762829aabe786a6f1acd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa8f6222620e848633e07558c8fd7101
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fcf38b0c107da901d39def7e75bab947(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37d5ed89d17fec89865171b50b0c5287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcf38b0c107da901d39def7e75bab947
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f1211eb27f7c5e640b1825307ca6765(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 8, 8, 7, 7, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c6a98a47382f3f693eacb6f0932fb3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f1211eb27f7c5e640b1825307ca6765
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74fa8e696a33627e69864527fcc84da5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[72, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bdaa6ca66cb4e8a1c17966f8a308f8b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74fa8e696a33627e69864527fcc84da5
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4deb00e35ce3d2dbd84eb0d082b93f3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
                paddle.static.InputSpec(shape=[18, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c17eb07f6cff849d66e2783f80b3acfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4deb00e35ce3d2dbd84eb0d082b93f3d
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.626307010650635, 4.391082763671875, 4.806268215179443, 4.401785850524902, 4.369851112365723, 4.688978672027588, 4.786715984344482, 4.2709174156188965, 4.522144794464111, 4.13695764541626, 5.095004558563232, 4.644773006439209, 3.8188412189483643, 4.205009937286377, 4.518152236938477, 4.431371212005615, 4.864565849304199, 3.6953768730163574]], dtype='float32').reshape([1, 18]),
                paddle.uniform([18, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6810a33b5889360a29b26f153fb6a3e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_550fcf5e4693870fd7b51093989a65e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6810a33b5889360a29b26f153fb6a3e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_89db5ef93c36f23fd18ec0419b473cb1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a77c6343c7b1e94121df49b25055cd9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89db5ef93c36f23fd18ec0419b473cb1
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_02e2f71ed5b4ff7a77955e6668f72d5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce70493bd360bdff0685923fba840621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02e2f71ed5b4ff7a77955e6668f72d5c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_361753ea5cf2b598e2c7ebbebd884ab7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e4b25c24b1fc1ac437bf4a1075ae7d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_361753ea5cf2b598e2c7ebbebd884ab7
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0dd73cea77706aaa848eec5e572be8bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[92, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_95fe5639a8dfd10f1cf43f9bbe892339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dd73cea77706aaa848eec5e572be8bc
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.uniform([92, 23], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea5ba5b1dfdb7bfe1726f623b0b9999f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
                paddle.static.InputSpec(shape=[23, 92], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5fa8290cab5b13d69e3e5bd99e030ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea5ba5b1dfdb7bfe1726f623b0b9999f
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.27309513092041, 5.833820343017578, 5.503009796142578, 5.4817891120910645, 5.425642013549805, 6.097849369049072, 5.734645843505859, 5.922000885009766, 5.353610038757324, 5.361513137817383, 4.979347229003906, 6.077544212341309, 5.27100133895874, 5.504151821136475, 5.766942977905273, 5.257235527038574, 5.357554912567139, 5.529504299163818, 5.592182159423828, 5.393463611602783, 5.493752479553223, 6.035822868347168, 5.610565185546875]], dtype='float32').reshape([1, 23]),
                paddle.uniform([23, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e775d153f77e848ef816d434c7c98d08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_247c9751a799e8ce95345a1dcc30e443(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e775d153f77e848ef816d434c7c98d08
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f907a5be0b6fbc8a71cb25882be5d33c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 3, 64, 198], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c101ae68bde89f2f7ebfe5a75a95fcc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f907a5be0b6fbc8a71cb25882be5d33c
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f490ff51bb856f6bed67bc2f949a2e52(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 198, 198], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d51a439854b8a8a95c9024c79abbcfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f490ff51bb856f6bed67bc2f949a2e52
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45ee19a834cde38fb4845850bfabec24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 198, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d520a34e54b74bf1be093325e46de65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45ee19a834cde38fb4845850bfabec24
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d33166154a81ebd7c7bc5fdd132125e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0022596490166c51479e70d338b98b9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d33166154a81ebd7c7bc5fdd132125e2
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2e84467bad44ab5f6227cf3533bea20f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ffb9395a95d2bed578647a30fbf55599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e84467bad44ab5f6227cf3533bea20f
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fdc395985aab29433aff0055df32049(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4be8a48852c274675cef1a59aaf088b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fdc395985aab29433aff0055df32049
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d047d98c09527ee0c00d0dfd8827be36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa8f6222620e848633e07558c8fd7101
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f12a370e29a2f403252822c54f969dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_766518ba1fdff3b0d8f67093cc25b0c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f12a370e29a2f403252822c54f969dc
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f0f4b6a1e0b08e81ae71b72825aedf6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[960, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e70c0a7b510f0e5d51d6d9276e4738e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f0f4b6a1e0b08e81ae71b72825aedf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([960, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_96e5df5a3f466efdfc382486697041c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 960], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a684f2dccddd8682233c53dc629743ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96e5df5a3f466efdfc382486697041c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 960], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9f76a81b51ccc737fe4784741be13cfd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c27475fbf5adb9fc152d3166348c9a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f76a81b51ccc737fe4784741be13cfd
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95a68789afe337e4aba2c406ef5fdcb9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9632b3acdea06b860ffdd0b1c61bf80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95a68789afe337e4aba2c406ef5fdcb9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9234460e4310bd96154c24217be5f70f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[12544, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a64254557a8c045cc1365724fa6cce55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9234460e4310bd96154c24217be5f70f
        def get_inputs(self):
            return [
                paddle.uniform([512, 12544], dtype='float32', min=0, max=0.5),
                paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c800759ba80a79d0c13335561264cf61(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4248991237606b95c5604dc5876fb04f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c800759ba80a79d0c13335561264cf61
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2db299162fb280d87510671fb5d34b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ba6d10efa8d63ad4655c2e114a1629a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2db299162fb280d87510671fb5d34b4
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48a5dd1a8c2268926baf5349224a44e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
                paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27d88029f5dc2cfaf8fa6308c5832384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a5dd1a8c2268926baf5349224a44e5
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_67d5bb8524e6ba72572e3e5489790f39(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 8, 8, 7, 7, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb1178f6e69ecb9790f34dd94f203b0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67d5bb8524e6ba72572e3e5489790f39
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_43cc4bb746b86de4961a684085def11f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f9cb823cd37470b468de156c60299950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43cc4bb746b86de4961a684085def11f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f1adcd94a73f1af7a12eb34dfb8e140(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43cc4bb746b86de4961a684085def11f
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07040b517a1ae60938156d75d10a0c1c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8061e86ce43e12a56b9f53b715739136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07040b517a1ae60938156d75d10a0c1c
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ee86c3226938732a3273c1a472b52af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f76bc47277dd585b01bed35a4058ad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ee86c3226938732a3273c1a472b52af
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5a8a0fe7f66d6e33b03de7d9bd4f4437(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9bf1b0168fe2ccfdd32d2877a4e3d4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a8a0fe7f66d6e33b03de7d9bd4f4437
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e26bc16e7b2e1335a16b9ff53f10730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2db299162fb280d87510671fb5d34b4
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_213b69b0125ba91c122d21e3099240d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a5dd1a8c2268926baf5349224a44e5
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8080cbf2c589829f5dded0059b2d709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fdc395985aab29433aff0055df32049
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e26bc16e7b2e1335a16b9ff53f10730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2db299162fb280d87510671fb5d34b4
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_213b69b0125ba91c122d21e3099240d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a5dd1a8c2268926baf5349224a44e5
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_41fe10e81f666d37a203ff64b3737b23(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d460b862572041046b026f59922824d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41fe10e81f666d37a203ff64b3737b23
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_632025868c55048f38a46fccd437af11(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f149edf44b57c0e4f1664c4307600fbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_632025868c55048f38a46fccd437af11
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_021dfb5c224c7ac3fbba6446ae7ee96f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6810a33b5889360a29b26f153fb6a3e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_417763a58b4714c66b27f403d5ff17ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89db5ef93c36f23fd18ec0419b473cb1
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aed90bde708af7112d2f4c1772461671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02e2f71ed5b4ff7a77955e6668f72d5c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeeea167fe79bd930a2fda936390fe13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_361753ea5cf2b598e2c7ebbebd884ab7
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_361d7edae5d866d85f809522b84797cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_188b1962cf6e8b4cb1a557f4a9d4891e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_361d7edae5d866d85f809522b84797cc
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_796fba39eb3bc2f061631e3c9c65b785(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1826ca0ffa4e58f7aaba91f8c8a82f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_796fba39eb3bc2f061631e3c9c65b785
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_787c99fb19c95946e367ae8df64e2f62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 4, 4, 7, 7, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72183e5c2ca039d71d8c47892925fbc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_787c99fb19c95946e367ae8df64e2f62
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_70e107f6e72a500bac44b91aff1597c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_538493b613fd91f321ddaf58e6709997(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70e107f6e72a500bac44b91aff1597c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9c6a5d2a1b1b715dd62538d7793c3a89(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0aa418c4151b0f529ca2a43e47da64e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6a5d2a1b1b715dd62538d7793c3a89
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 577], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_83e8309de34406c7ffa5553848e77b2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e1ae8ab1558ca50be1a5510e497ee23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83e8309de34406c7ffa5553848e77b2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15773e622c255cdb809185e9a046da31(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef85df8317e4ffde3ec3957d9ca8080b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15773e622c255cdb809185e9a046da31
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cc12c9a810d6724e64c58e00577712e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ee86c3226938732a3273c1a472b52af
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0922028c906e1392d51fbf2892560158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a8a0fe7f66d6e33b03de7d9bd4f4437
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_adedf1b94cda19bc52f0d25678173d3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4951e4a66875b4177ef0fec2d52103bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adedf1b94cda19bc52f0d25678173d3d
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb1642d8f8009d96db1f6307bb206f63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43cc4bb746b86de4961a684085def11f
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ae0794268a0b1420e72ef383b573142b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[872, 218], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb49929d9b44795f223c4781f28aa371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae0794268a0b1420e72ef383b573142b
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_556d8684aadb1e18d35a40053dc5aaf7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
                paddle.static.InputSpec(shape=[218, 872], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89d61bd12e0d4610401271fb8ac3abc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_556d8684aadb1e18d35a40053dc5aaf7
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d491283d35911575417543ce8789ec9d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 4, 4, 7, 7, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83d5174cbbb07db08d46e46c23e43bad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d491283d35911575417543ce8789ec9d
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4ff7f1cd493d085e32d5033be66b44ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d63f7e82bb2ef9214b29dd18465ae7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ff7f1cd493d085e32d5033be66b44ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23508cb682bd7465728d1472f543a43f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fdc22b330d978595db5b1dcc361cf67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23508cb682bd7465728d1472f543a43f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e3930ddc95dcc12fbe1ff1a0bc4300f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b444485abaf5b5ce2885fbe64700507(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3930ddc95dcc12fbe1ff1a0bc4300f1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7dbc662cf493a4dfb20b5ce14a98fab7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ba963f3e5ad7e6ab570ac6d1ea9af71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dbc662cf493a4dfb20b5ce14a98fab7
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d63f7e82bb2ef9214b29dd18465ae7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ff7f1cd493d085e32d5033be66b44ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d460b862572041046b026f59922824d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41fe10e81f666d37a203ff64b3737b23
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f149edf44b57c0e4f1664c4307600fbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_632025868c55048f38a46fccd437af11
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d2bed0a788e35881ec313788b6aa6d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42c3f018268faf21fe7cf16a2d625437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d2bed0a788e35881ec313788b6aa6d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a42571da0b7c796bd631f74256bdf7d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_40516f2ac45309b7504859627afab67a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a42571da0b7c796bd631f74256bdf7d
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e29426a102a9c646a9e61120cf2a4197(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[3136, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0a0619b78703dae91cbaf81f5063493(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e29426a102a9c646a9e61120cf2a4197
        def get_inputs(self):
            return [
                paddle.uniform([390, 3136], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0326fb28cd6193b1cdc21b8182588b7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c800759ba80a79d0c13335561264cf61
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45beee0928411e95b8ebb65bf1283f17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2db299162fb280d87510671fb5d34b4
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9833e295846508d0695dd7d2fa8e1ed3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a5dd1a8c2268926baf5349224a44e5
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ab62d2f26f0049152cfc045a4b65e6fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2eb1a484d5bce7ea12fa52b836d40bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab62d2f26f0049152cfc045a4b65e6fa
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2551ab3d8e7fcaff96cc2e5df55ceb88(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbc4510ad0fdac04b372cfdee901d004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2551ab3d8e7fcaff96cc2e5df55ceb88
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6fe77c890c7c3edef938793121b40030(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f9be8532c41ae186115ef9bb5403a5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe77c890c7c3edef938793121b40030
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6ef2b914abfb5c7c28881ba283574a92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92dd3a2e071a0bc449b44f1166736509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ef2b914abfb5c7c28881ba283574a92
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d582d9138c8848c572ffcee5b7b35bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_361d7edae5d866d85f809522b84797cc
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afa3eedba4f81715a3271fc6e30510f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_796fba39eb3bc2f061631e3c9c65b785
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7456ccb87189dcb69da1813d645da0a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e775d153f77e848ef816d434c7c98d08
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_904cde6c5086fd40a8d0fb9694d6de7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f907a5be0b6fbc8a71cb25882be5d33c
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c874b842d1acef08e4a202733bf9600(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f490ff51bb856f6bed67bc2f949a2e52
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b45b933866f61b17406507260fe2fae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45ee19a834cde38fb4845850bfabec24
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3698a3df0b9c00d5729ecc79e730561b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41fe10e81f666d37a203ff64b3737b23
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17b474325fd7b61c4cc3aa6dedd346e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2263dac67a6ac445a106868278979f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b474325fd7b61c4cc3aa6dedd346e3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15c3310ac2b7e8f41babc61b7af23569(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b0af861d3bafbeb98e99fb6769bd45a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c3310ac2b7e8f41babc61b7af23569
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b0af861d3bafbeb98e99fb6769bd45a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c3310ac2b7e8f41babc61b7af23569
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4e35dc8f6d9d9749f22d13b94233420a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb7e1784937fd5f819c05848c1e15617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e35dc8f6d9d9749f22d13b94233420a
        def get_inputs(self):
            return [
                paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96e805b13dfa0ac2bf3e854435df7105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c3310ac2b7e8f41babc61b7af23569
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96e805b13dfa0ac2bf3e854435df7105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15c3310ac2b7e8f41babc61b7af23569
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_93aa7cd88fe241ae3df7af8f086696ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 2048], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b06139696e4d5b948d609ffc1f8f08f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93aa7cd88fe241ae3df7af8f086696ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 2048], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a4ddd48dbe1c160264cbb7711833546(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2048], dtype='float32'),
                paddle.static.InputSpec(shape=[2048, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed35a9ea6d12e57661a09bd1400ae24c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ddd48dbe1c160264cbb7711833546
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f85e6c5e22a1c5fbc95feb4d88b2f20(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1, 1, 7, 7, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25fb9d2c37e308d50756373c5728a73a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f85e6c5e22a1c5fbc95feb4d88b2f20
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a7b3db68c008baef9a666fabd7888a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d33166154a81ebd7c7bc5fdd132125e2
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7edc48b80b45cf195dc21ca2c01de48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e84467bad44ab5f6227cf3533bea20f
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ba6d10efa8d63ad4655c2e114a1629a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2db299162fb280d87510671fb5d34b4
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27d88029f5dc2cfaf8fa6308c5832384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a5dd1a8c2268926baf5349224a44e5
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb1178f6e69ecb9790f34dd94f203b0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67d5bb8524e6ba72572e3e5489790f39
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59f015c54bb1baac3a1998ac549f37b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fdc395985aab29433aff0055df32049
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c6d23e411c8cb3b2d9ef3162df2d165a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78b594030d12924224608075272b668d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d23e411c8cb3b2d9ef3162df2d165a
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1412c7057fd66866aa352183a2521a99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ddc345f1e9771eeb7cce159dffaccf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1412c7057fd66866aa352183a2521a99
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8806ca99e00098f16337ccc75431e8c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b7e5198c507545f923c0e78ef1bbe9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8806ca99e00098f16337ccc75431e8c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_10c8966697e6cc107776becf6c152c43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5709e442287239ec6cb0adf6247d10c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_10c8966697e6cc107776becf6c152c43
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8b68da9f819b378db009a4635f03daee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_450ad709e720a5c86d09d31e2c06d8e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b68da9f819b378db009a4635f03daee
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b7ba7543fa6b16563bbaba1d8bcfcc2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f253ce29f22b303b071a9a853ddb187e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b7ba7543fa6b16563bbaba1d8bcfcc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b7e5198c507545f923c0e78ef1bbe9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8806ca99e00098f16337ccc75431e8c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9440035eaf45f48dda0ed3b3584f4609(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b5815956d484a388c64301352f29ea5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9440035eaf45f48dda0ed3b3584f4609
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1658b6e8ca6f23a806db71b3c9a0828e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43cc4bb746b86de4961a684085def11f
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e279bd219c149a8523b6137b56abacf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4b44e4b1ec0ee5fcf260032a30264329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e279bd219c149a8523b6137b56abacf
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_550fcf5e4693870fd7b51093989a65e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6810a33b5889360a29b26f153fb6a3e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a77c6343c7b1e94121df49b25055cd9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89db5ef93c36f23fd18ec0419b473cb1
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce70493bd360bdff0685923fba840621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02e2f71ed5b4ff7a77955e6668f72d5c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e4b25c24b1fc1ac437bf4a1075ae7d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_361753ea5cf2b598e2c7ebbebd884ab7
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c3b8bf21a3fa0645207e0d15fb691a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f76a81b51ccc737fe4784741be13cfd
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3afa6e160b8dc73f675e31b931129644(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95a68789afe337e4aba2c406ef5fdcb9
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ccc71b75a4a185baf4288c25e2b808b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70e107f6e72a500bac44b91aff1597c1
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2facca1f4e82859eae3d550131ee59c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2db299162fb280d87510671fb5d34b4
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2e0d5bbcfb62c294d5e71f8b381a901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a5dd1a8c2268926baf5349224a44e5
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73ae3929c6287702db5fd575eff148d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_361d7edae5d866d85f809522b84797cc
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3b1c7170fea92f92b827b786d25ae21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_796fba39eb3bc2f061631e3c9c65b785
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45beee0928411e95b8ebb65bf1283f17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2db299162fb280d87510671fb5d34b4
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9833e295846508d0695dd7d2fa8e1ed3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a5dd1a8c2268926baf5349224a44e5
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9d007f7aacde0c7988bd648f4162a3c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9fb66b05ed9209740ca1a268ec6bf258(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d007f7aacde0c7988bd648f4162a3c9
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14038e050d00c821938857fc4908c157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ee86c3226938732a3273c1a472b52af
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea613af3bad85746b0ffd3df6e55a7a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a8a0fe7f66d6e33b03de7d9bd4f4437
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01e35815d121576863dbcb563c6ddb6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_361d7edae5d866d85f809522b84797cc
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04133d109414d8d93e73fdb6ce255726(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_796fba39eb3bc2f061631e3c9c65b785
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab1a12fad01b050add28df1bb0a53e9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d007f7aacde0c7988bd648f4162a3c9
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83d5174cbbb07db08d46e46c23e43bad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d491283d35911575417543ce8789ec9d
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4be8a48852c274675cef1a59aaf088b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fdc395985aab29433aff0055df32049
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_51ad3848569e6916a125da4ec210791f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e01a6c8ebcdd5cb9bc3f997b154f3c29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ad3848569e6916a125da4ec210791f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_972bf7ba2b7fafaa67bf64e9740ac5ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 6, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d4e72d356694f50f54f069f6a90659b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_972bf7ba2b7fafaa67bf64e9740ac5ac
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_14283984a5d20c65152347ac9b3b4210(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_367dd1d7d1350eec3f12c4f648a30dab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14283984a5d20c65152347ac9b3b4210
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_39cba05f84a06e02955001fa793b0dc6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b9fd0c382182695294fa2181465e59e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39cba05f84a06e02955001fa793b0dc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dfeb7ff1a1f00c092e79dfc43b7b2c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d23e411c8cb3b2d9ef3162df2d165a
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_184e3f658245b7e377136e2a5eac97ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1412c7057fd66866aa352183a2521a99
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_965dc475a5d7bce16c8efc9eb237b27c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d2bed0a788e35881ec313788b6aa6d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fafa435c6f3ac8be69a6f1197f00b79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a42571da0b7c796bd631f74256bdf7d
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d182820710a5ff87b6e31ff44fc7ff43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 150], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1f714f222ccae55513b6720a747e287b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d182820710a5ff87b6e31ff44fc7ff43
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_febd5cf1c087e1b346a59a8941e43203(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ad3848569e6916a125da4ec210791f
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_902333d3c7811aeb28f45feac5e3a394(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ee86c3226938732a3273c1a472b52af
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e78e0210ea55c12724cd92194ac9e82b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a8a0fe7f66d6e33b03de7d9bd4f4437
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_132dbcacf43183750daabc8f5633d8ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[672, 168], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7af912581a33b9f412b73082b1ad60b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_132dbcacf43183750daabc8f5633d8ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a70612b1e4535666e5a62ebec79c0a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
                paddle.static.InputSpec(shape=[168, 672], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab673da0c1a353ae2dbc6bf6bc1c2af0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a70612b1e4535666e5a62ebec79c0a7
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2b03f2a8c6217147961f891278adb558(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae1dc3ffc27f3c09cec0fa2459fee654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b03f2a8c6217147961f891278adb558
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a61091275560d7c12f2510db4438ccfd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ad3acdfb41d44126954b2bdb35908f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a61091275560d7c12f2510db4438ccfd
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3d72966c8a4aa4ee9376f8e67ba0db0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e64e324d671b44b24ef3a38b7cf8af2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3d72966c8a4aa4ee9376f8e67ba0db0
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e3e30c4abbcbfa6f2e8c12cca35858f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5e4b632b392f1ece73d7460c72e88d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e3e30c4abbcbfa6f2e8c12cca35858f
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae1dc3ffc27f3c09cec0fa2459fee654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b03f2a8c6217147961f891278adb558
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_217dd21f4295a9858c68badf52ddb1ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8806ca99e00098f16337ccc75431e8c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a3ed2a7910e96404643f7b4b4ade15e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_10c8966697e6cc107776becf6c152c43
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bc1c2bbfd488c87a10142f2e3cde279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b68da9f819b378db009a4635f03daee
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d596af5c2b4c4b81c1ecf89de6cd9e82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b7ba7543fa6b16563bbaba1d8bcfcc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_217dd21f4295a9858c68badf52ddb1ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8806ca99e00098f16337ccc75431e8c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2facca1f4e82859eae3d550131ee59c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2db299162fb280d87510671fb5d34b4
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2e0d5bbcfb62c294d5e71f8b381a901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48a5dd1a8c2268926baf5349224a44e5
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7885c31a20c8762829aabe786a6f1acd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa8f6222620e848633e07558c8fd7101
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37d5ed89d17fec89865171b50b0c5287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcf38b0c107da901d39def7e75bab947
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ada2dd466cb0fc07b8054da649bc454(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fdc395985aab29433aff0055df32049
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4be8a48852c274675cef1a59aaf088b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fdc395985aab29433aff0055df32049
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_14ec9053ff1c5849d3e346a2cf328895(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9511639f2d25e9e0dd628bffa2d70538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14ec9053ff1c5849d3e346a2cf328895
        def get_inputs(self):
            return [
                paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25fb9d2c37e308d50756373c5728a73a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f85e6c5e22a1c5fbc95feb4d88b2f20
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_993d9dd279c227bed4a9f05e22c26f7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e775d153f77e848ef816d434c7c98d08
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1e5542d62a2264ebfc40e513becc2439(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 3, 64, 197], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a213c621e8f8d96a59505ba440e4999(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e5542d62a2264ebfc40e513becc2439
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_736e5fd1f9f7669b005f1cf5169ef694(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 197, 197], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88769c2b082c655a63da2c3049547292(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_736e5fd1f9f7669b005f1cf5169ef694
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5a36058ea4f1a77b0dbc0c0a9caa6a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 197, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9bc7d238428522b27629098b0b813489(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a36058ea4f1a77b0dbc0c0a9caa6a0
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87d1efc1c6d891ff3301363c838acb04(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81b27a43b38fbe2b3458f69c5967c5d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87d1efc1c6d891ff3301363c838acb04
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0089cace56d259a33f8eccb00e630189(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_446c313bc54d6a7598c60709fd6e52f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089cace56d259a33f8eccb00e630189
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5319deb59cc433779da12e8af94acb7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0bc208462158356194b98cbcffa8bf03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5319deb59cc433779da12e8af94acb7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e7a044bbb944b0e2efbd8ef5caf4a674(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e6bf3e07a833aade55681ed3c337a2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7a044bbb944b0e2efbd8ef5caf4a674
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81b27a43b38fbe2b3458f69c5967c5d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87d1efc1c6d891ff3301363c838acb04
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48f8f2393c2685a7100703d6847ff7bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e775d153f77e848ef816d434c7c98d08
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72183e5c2ca039d71d8c47892925fbc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_787c99fb19c95946e367ae8df64e2f62
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3698a3df0b9c00d5729ecc79e730561b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41fe10e81f666d37a203ff64b3737b23
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2263dac67a6ac445a106868278979f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b474325fd7b61c4cc3aa6dedd346e3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fe9690de5dff4942110cafb27a081a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 6625], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd9fb0609f9729606d6579b9d23d641f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fe9690de5dff4942110cafb27a081a7
        def get_inputs(self):
            return [
                paddle.uniform([10, 40, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 6625], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_021dfb5c224c7ac3fbba6446ae7ee96f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6810a33b5889360a29b26f153fb6a3e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_417763a58b4714c66b27f403d5ff17ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89db5ef93c36f23fd18ec0419b473cb1
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aed90bde708af7112d2f4c1772461671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02e2f71ed5b4ff7a77955e6668f72d5c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeeea167fe79bd930a2fda936390fe13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_361753ea5cf2b598e2c7ebbebd884ab7
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3170531079b29050f8dee339fbfc17b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07ca628ef4e0b507fdc994ac65094013(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3170531079b29050f8dee339fbfc17b3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf4dbaa64b2403493405561940b83fad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_199501b898552ef8a5bee58743220769(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf4dbaa64b2403493405561940b83fad
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c232f9224c3f9d2a534ae5f3303044d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a764931b5c3b3557fa66b8207fbc5a68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c232f9224c3f9d2a534ae5f3303044d8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9c1466926726cd6692be58ee854a722f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99865485087a2c60e9524b3d3eb1bdd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c1466926726cd6692be58ee854a722f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07ca628ef4e0b507fdc994ac65094013(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3170531079b29050f8dee339fbfc17b3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_965dc475a5d7bce16c8efc9eb237b27c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d2bed0a788e35881ec313788b6aa6d7
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fafa435c6f3ac8be69a6f1197f00b79b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a42571da0b7c796bd631f74256bdf7d
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b59525f41035eae3778e6f38efc550ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 150], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2e1fb213662326852a36829499030e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b59525f41035eae3778e6f38efc550ef
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f779fb3733cce3fb95dcdd97e78e0192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab62d2f26f0049152cfc045a4b65e6fa
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d26cd1f96cd743d7c8a94eea9ccf2dc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2551ab3d8e7fcaff96cc2e5df55ceb88
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ce83f715cebf83309068be89dfe72a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fe77c890c7c3edef938793121b40030
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1fb389c4ce1f34f6a1db10a58e01be5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ef2b914abfb5c7c28881ba283574a92
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e714b0185c5ea1dfb27fbac9ce213d91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70e107f6e72a500bac44b91aff1597c1
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59c8d54ceaf1477e03a3cdeab9c12e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_adedf1b94cda19bc52f0d25678173d3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81d1fcadeaab9b1e7425f3561018bcda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43cc4bb746b86de4961a684085def11f
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f71edb2229db75839c660170be6769fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 2, 2, 7, 7, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bea6c9d8a006391d1a636e0ca43d709c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f71edb2229db75839c660170be6769fb
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c63e8c715f82ba7ee01ca9bbeed48072(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3c855924d24791950e43a4f2cf32e151(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c63e8c715f82ba7ee01ca9bbeed48072
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e453c3289d1a73ba16df2c348a6fe81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23508cb682bd7465728d1472f543a43f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_615f2a00cc9bf633971441a2760dbace(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32de000c0cd7e0b854c70ee6fc3d4b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_615f2a00cc9bf633971441a2760dbace
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_68256d02777452a6d5a1fbfe887e29b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d98461ce19201106a74ca082b75f5f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68256d02777452a6d5a1fbfe887e29b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c855924d24791950e43a4f2cf32e151(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c63e8c715f82ba7ee01ca9bbeed48072
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a152f4c3b0a9721b7394f57c779c650(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[704, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1f9e914882a6ddea89939ff7c97ca91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a152f4c3b0a9721b7394f57c779c650
        def get_inputs(self):
            return [
                paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f08af081a1da3c181690cee7d4f746aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe15ba72573c8c007f2635aa1ede0813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f08af081a1da3c181690cee7d4f746aa
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dd452fe6062676d3d3977ca6c6fc8dc0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 640], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a56188212bf5b534c20c575a40e19c13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd452fe6062676d3d3977ca6c6fc8dc0
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a95ce6339822e8dacbcf7479ce3c45b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3826303150dbc7abc8db683abfe1791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a95ce6339822e8dacbcf7479ce3c45b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_63ec50311035d2b36ee7cab1fa22c421(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f74558c4161f0641acd96eea6bcbf1a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63ec50311035d2b36ee7cab1fa22c421
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe15ba72573c8c007f2635aa1ede0813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f08af081a1da3c181690cee7d4f746aa
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9b4205033238d1759763b301be3ab4e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1248, 312], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5a0c2dac7ab95de36f1d7444b310bcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b4205033238d1759763b301be3ab4e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.uniform([1248, 312], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5b76ba1059891194099ea43e39336795(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
                paddle.static.InputSpec(shape=[312, 1248], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3e9b17f4c64b92325e3272bf57a901c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b76ba1059891194099ea43e39336795
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
                paddle.uniform([312, 1248], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_898f0ba486e7324c6e7041cb687535fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56880154e0df1381f4b64d8c6e5c6559(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_898f0ba486e7324c6e7041cb687535fd
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c18c4d389f701f49f0c13d78b27758be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f76a81b51ccc737fe4784741be13cfd
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d2df5f9f1ccfa814652affccb398d23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95a68789afe337e4aba2c406ef5fdcb9
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97977fa13060dbc256af8746b62dea14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d23e411c8cb3b2d9ef3162df2d165a
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf79b0b09e9da913198ecc79665631b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1412c7057fd66866aa352183a2521a99
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_490e1a283278b7099ac0ad4bfd241604(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1, 1, 7, 7, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60332e84d019192def985e3d47f7638b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_490e1a283278b7099ac0ad4bfd241604
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ed73457c1949b89b342392a288def26a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 2, 2, 7, 7, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9d0973c3003a0de6df2c019b81a0d82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed73457c1949b89b342392a288def26a
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60332e84d019192def985e3d47f7638b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_490e1a283278b7099ac0ad4bfd241604
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53da1be927a5798079fbeba1980e1830(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70e107f6e72a500bac44b91aff1597c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_871ea04de778ed69138a85f58810861a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6a5d2a1b1b715dd62538d7793c3a89
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df3a839e8062c53026845ea30e9b5cb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83e8309de34406c7ffa5553848e77b2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2741f35183e8a226446e4e162d4cc6e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15773e622c255cdb809185e9a046da31
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9d0973c3003a0de6df2c019b81a0d82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed73457c1949b89b342392a288def26a
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_926b8c42c592123ad85ca60f5db0adfd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[156, 39], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64ed299b54ce63e39a12f5f8ed53991d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_926b8c42c592123ad85ca60f5db0adfd
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 39], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea15ea8c1c43f17fc5450317174bf0ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
                paddle.static.InputSpec(shape=[39, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2eea48a0578d6e67ee22b45986d1d4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea15ea8c1c43f17fc5450317174bf0ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
                paddle.uniform([39, 156], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34730154d6de41314bad2a3f63c4bc30(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_921bccf9208a3488f1d3d004ce024d4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34730154d6de41314bad2a3f63c4bc30
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d9fb96f24598a13d4945af461094e6f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbcf390f6ceb8282bcb817ca6ac44935(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9fb96f24598a13d4945af461094e6f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_516541da2cf86b8b12c2264f9cde94f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2591f26609f84ea29979f3993ac7c7dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_516541da2cf86b8b12c2264f9cde94f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0444b0b54b9a9466d50f75b882c10f86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27e5c76f132c400b143520de180a90c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0444b0b54b9a9466d50f75b882c10f86
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_921bccf9208a3488f1d3d004ce024d4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34730154d6de41314bad2a3f63c4bc30
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb49929d9b44795f223c4781f28aa371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae0794268a0b1420e72ef383b573142b
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89d61bd12e0d4610401271fb8ac3abc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_556d8684aadb1e18d35a40053dc5aaf7
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c6a98a47382f3f693eacb6f0932fb3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f1211eb27f7c5e640b1825307ca6765
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bea6c9d8a006391d1a636e0ca43d709c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f71edb2229db75839c660170be6769fb
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e95be146855e1acba243f152c26ddec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_898f0ba486e7324c6e7041cb687535fd
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_095a84f2494c66f7bfa1646c20da1b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f76a81b51ccc737fe4784741be13cfd
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3a7b6baa540dce20bf764dd22718cb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95a68789afe337e4aba2c406ef5fdcb9
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_843f7e314be335d66399e461a8fe1924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f76a81b51ccc737fe4784741be13cfd
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8e3835dd0f7a08443e700566f1df5d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95a68789afe337e4aba2c406ef5fdcb9
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42c3f018268faf21fe7cf16a2d625437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d2bed0a788e35881ec313788b6aa6d7
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40516f2ac45309b7504859627afab67a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a42571da0b7c796bd631f74256bdf7d
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_616e75b1146247b2bdc028b8e9cdb14f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 37], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0be13e1324b41dcab9c646813a77a3c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_616e75b1146247b2bdc028b8e9cdb14f
        def get_inputs(self):
            return [
                paddle.uniform([10, 25, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 37], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d6198ae60c716ba2a3af0a8d137e20e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d23e411c8cb3b2d9ef3162df2d165a
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9cb0397822f31664acc37bd4e5300d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1412c7057fd66866aa352183a2521a99
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_97767d43222693ee82e87facb4088082(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_340086b3d71c5f394fe4a3fc62dfdc18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97767d43222693ee82e87facb4088082
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4bf906fb9075d1199436dc30dc2708b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[30, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab46f728a7a1648ebcad5f2f61e7d860(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4bf906fb9075d1199436dc30dc2708b
        def get_inputs(self):
            return [
                paddle.to_tensor([[8.870004653930664, 9.070971488952637, 8.6998291015625, 8.771418571472168, 8.77302074432373, 8.278258323669434, 9.320741653442383, 8.51421070098877, 8.791091918945312, 8.457852363586426, 9.51916790008545, 8.142207145690918, 8.19686508178711, 8.17872428894043, 8.666762351989746, 8.191591262817383, 9.497175216674805, 8.180706024169922, 8.71663761138916, 9.759553909301758, 8.672922134399414, 8.273152351379395, 8.350447654724121, 9.068215370178223, 8.876349449157715, 8.625724792480469, 9.934979438781738, 8.560579299926758, 8.065917015075684, 9.131192207336426]], dtype='float32').reshape([1, 30]),
                paddle.uniform([30, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4166d6a0c790e921533ec9c342f3ca69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f8090631f1b66ca29f2043b79956053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4166d6a0c790e921533ec9c342f3ca69
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cafbf7a7621957bfc8ca2ef8ee86d504(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd452fe6062676d3d3977ca6c6fc8dc0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be1207e9f43961afc3b2725fdfad3f90(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f95533fec7bdd08ed68efcaa1d4f49db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be1207e9f43961afc3b2725fdfad3f90
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_246525f70a0550b9203cebd9ba940bbc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cadd5b1c812c05953a6381608f6009e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_246525f70a0550b9203cebd9ba940bbc
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f8090631f1b66ca29f2043b79956053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4166d6a0c790e921533ec9c342f3ca69
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d7718206a77b312929288ac75bb98c50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8bf7ab668f482a7f1427453fa04985ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7718206a77b312929288ac75bb98c50
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd4ed059838de78cbef2c9de87428151(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a61091275560d7c12f2510db4438ccfd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c795b97c3cdc36e99a5e50d6bf44970(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a6285438d4335c57585ce17b44d3dba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c795b97c3cdc36e99a5e50d6bf44970
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e7104aade60a1cd9ecf24c10dc0f29f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_26000066f378eca7694f49da64269f1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7104aade60a1cd9ecf24c10dc0f29f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bf7ab668f482a7f1427453fa04985ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7718206a77b312929288ac75bb98c50
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7af912581a33b9f412b73082b1ad60b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_132dbcacf43183750daabc8f5633d8ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab673da0c1a353ae2dbc6bf6bc1c2af0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a70612b1e4535666e5a62ebec79c0a7
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f1adcd94a73f1af7a12eb34dfb8e140(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43cc4bb746b86de4961a684085def11f
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8061e86ce43e12a56b9f53b715739136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07040b517a1ae60938156d75d10a0c1c
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ecd9d92e91b171dc597794d72e447d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34730154d6de41314bad2a3f63c4bc30
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8329b26892f36f881125b16c3e9392ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9fb96f24598a13d4945af461094e6f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08704ac9a602532a01e85f3e51f21a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_516541da2cf86b8b12c2264f9cde94f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60a77ec72a9e6d9d7fca3cace80b7535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0444b0b54b9a9466d50f75b882c10f86
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ecd9d92e91b171dc597794d72e447d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34730154d6de41314bad2a3f63c4bc30
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_317f38cecc3a642911727cc6a4869806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e775d153f77e848ef816d434c7c98d08
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bef31518a91b9741d2381b50042a9027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d2bed0a788e35881ec313788b6aa6d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d047d98c09527ee0c00d0dfd8827be36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa8f6222620e848633e07558c8fd7101
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_766518ba1fdff3b0d8f67093cc25b0c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f12a370e29a2f403252822c54f969dc
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1658b6e8ca6f23a806db71b3c9a0828e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43cc4bb746b86de4961a684085def11f
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b44e4b1ec0ee5fcf260032a30264329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e279bd219c149a8523b6137b56abacf
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_312a9c5053bcb38fbe8a9e0c59613825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ad3848569e6916a125da4ec210791f
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3cc3416bc95835ac2c7ec5c3525b7b71(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef069fb472fb3d1e0c0dccdca21d6b78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cc3416bc95835ac2c7ec5c3525b7b71
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a8c3390d570a8ec2c6e0a6d0df186d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf4dbaa64b2403493405561940b83fad
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_40eadad164a9a4c28de2ddff667dbedb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4e8cd12881ca5b04ed401a798b5d395(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40eadad164a9a4c28de2ddff667dbedb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e90fed732cb928489cc5a5ff9f96812(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7b7df2a93af7b2e3c277142c2abf5c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e90fed732cb928489cc5a5ff9f96812
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef069fb472fb3d1e0c0dccdca21d6b78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cc3416bc95835ac2c7ec5c3525b7b71
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa9d6122628f26554e02e86885f67773(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c28b157225c1cbab737c5fbf55604733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9d6122628f26554e02e86885f67773
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_199501b898552ef8a5bee58743220769(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf4dbaa64b2403493405561940b83fad
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5b287526cc825882422a4a9017ffa970(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_147ea08149ce607bee01c0a8398ed6f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b287526cc825882422a4a9017ffa970
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ca2cbd5cd496164699bdfb5cd93758b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad0a2f49955d218cf4ec0e16bdc9000c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca2cbd5cd496164699bdfb5cd93758b9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c28b157225c1cbab737c5fbf55604733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9d6122628f26554e02e86885f67773
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f10965a80c866b170115fab77275c256(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e775d153f77e848ef816d434c7c98d08
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc87620bf1be5c8d42566e3557b4eeca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e5542d62a2264ebfc40e513becc2439
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6055323ba7628073f584c076052d966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_736e5fd1f9f7669b005f1cf5169ef694
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74a1eb9d394bf1be90005c6738003457(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5a36058ea4f1a77b0dbc0c0a9caa6a0
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45292ea15aec1645c0b03a819c1c83f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a4e321986a9aef2baa7fcc63dcc26d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45292ea15aec1645c0b03a819c1c83f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cce999c7d7e25a0f6f3a20180cf7b08a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089cace56d259a33f8eccb00e630189
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1c5e0e3320231040cb4d7a4ca4acb774(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c292d16582b0bda7f98be3c6e0ee88d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c5e0e3320231040cb4d7a4ca4acb774
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_31c37e589c91237bcc5dec0763204616(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe5276cb5387355c97b4fefb97efbff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31c37e589c91237bcc5dec0763204616
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a4e321986a9aef2baa7fcc63dcc26d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45292ea15aec1645c0b03a819c1c83f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1cdd60dbe4eaa323765b7ec8fdbf96fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d38ce059400ec78369d1f0b3518fa19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cdd60dbe4eaa323765b7ec8fdbf96fc
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6dc46b9f4098eeb2027a06d6b52813a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b68da9f819b378db009a4635f03daee
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef48830b762377a380c0d7691615ce43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b7ba7543fa6b16563bbaba1d8bcfcc2
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37cfa00d746590efb5031819e987bf7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8806ca99e00098f16337ccc75431e8c1
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_441201762b53970c2153126e38f46ae4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51ad3848569e6916a125da4ec210791f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88526273f1e9cf05297349303ac1c951(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_972bf7ba2b7fafaa67bf64e9740ac5ac
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6076be4cf3411440ab23906788541259(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14283984a5d20c65152347ac9b3b4210
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bddb6c31d6439a0206351ac7f8e8a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39cba05f84a06e02955001fa793b0dc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d3cf4bc6b07b3673d3e24d114f7f1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9440035eaf45f48dda0ed3b3584f4609
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24c95d65da3a44dffda67e0581c0ca88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a152f4c3b0a9721b7394f57c779c650
        def get_inputs(self):
            return [
                paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab489d1e1de80ffedff9641dc9186436(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70e107f6e72a500bac44b91aff1597c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_680bca6d4ab17eac3131bdf04efa3f71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c6a5d2a1b1b715dd62538d7793c3a89
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_783d2fd2c2a3b78167c267e4d84b14f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83e8309de34406c7ffa5553848e77b2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85550895d7b39d221cd96dcf4df5c6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15773e622c255cdb809185e9a046da31
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a9c51780aaf8bcae65efc3352bff8129(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d805a5b49d4493986dedde245077e4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9c51780aaf8bcae65efc3352bff8129
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a8c3390d570a8ec2c6e0a6d0df186d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf4dbaa64b2403493405561940b83fad
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0716d3f2b51d03b56418423c3d186bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4434096634c4b6eda32a82ff5b8d3fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0716d3f2b51d03b56418423c3d186bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_55dc879d2f0395913d139d8ac26b8370(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0eb49481dc43b6e495b66de327736564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55dc879d2f0395913d139d8ac26b8370
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d805a5b49d4493986dedde245077e4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9c51780aaf8bcae65efc3352bff8129
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8e0687d92913a63bd86e3420b66eb383(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[624, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e55acd13493e73e7f52967a0f11f2ee9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e0687d92913a63bd86e3420b66eb383
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.uniform([624, 156], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07e5056b101eef2abfc27e5724b1ae85(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
                paddle.static.InputSpec(shape=[156, 624], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_676a5472f11819c0752d4646c2140d20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07e5056b101eef2abfc27e5724b1ae85
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5072ce4af629909be587c24a44d2ae6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cdd60dbe4eaa323765b7ec8fdbf96fc
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cc78c3d7555cf0323f4b2adc5cd33e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b68da9f819b378db009a4635f03daee
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d7bfc5266aed23a704857c685e390cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b7ba7543fa6b16563bbaba1d8bcfcc2
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38316da80ce03d3dc13aa84df6ae4e7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8806ca99e00098f16337ccc75431e8c1
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86522ead67d0b265d73ac93479a6c418(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
                paddle.static.InputSpec(shape=[91], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_883847a9bace7ba8e7bf8c3a7cbbaa0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86522ead67d0b265d73ac93479a6c418
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
                paddle.uniform([91], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c93c09ee72217202b86bd3e8c52a6d79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ceb9831b1f870488f77eb92c04013cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93c09ee72217202b86bd3e8c52a6d79
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37d5ed89d17fec89865171b50b0c5287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcf38b0c107da901d39def7e75bab947
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c6a98a47382f3f693eacb6f0932fb3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f1211eb27f7c5e640b1825307ca6765
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c2f6fef7298ebed734fa68117e47689(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[72, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c5b2f00ecb23bb9f6b48ed8110beb42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c2f6fef7298ebed734fa68117e47689
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5ec20d7b7c687993947acfb8e128b3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 18], dtype='float32'),
                paddle.static.InputSpec(shape=[18, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6f7cb6c5f1c41183e82fe173ddde0b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5ec20d7b7c687993947acfb8e128b3a
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.626307010650635, 4.391082763671875, 4.806268215179443, 4.401785850524902, 4.369851112365723, 4.688978672027588, 4.786715984344482, 4.2709174156188965, 4.522144794464111, 4.13695764541626, 5.095004558563232, 4.644773006439209, 3.8188412189483643, 4.205009937286377, 4.518152236938477, 4.431371212005615, 4.864565849304199, 3.6953768730163574]], dtype='float32').reshape([1, 18]),
                paddle.uniform([18, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c1a32489827fc616eb37c9af942cae24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10ff74d4cdfd0b9c3b5188fb89a7f631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a32489827fc616eb37c9af942cae24
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f528afedd4658a4504889b3f74223ff9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 4, 32, 100], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c6565e79ece1a1df6d2e76e6fb67efa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f528afedd4658a4504889b3f74223ff9
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7cf3a1855054150f51c85ab0accf41e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 100, 100], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 4, 100, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b08e2fd293b49f5b9e6e3714bada9949(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7cf3a1855054150f51c85ab0accf41e3
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4ef071c18706ab2be3f9c6c96756d18c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78e6cfcd4d29bc1c837e341e318c8a20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ef071c18706ab2be3f9c6c96756d18c
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bca3274da3af2c98c2af149f3c20cef4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 92], dtype='float32'),
                paddle.static.InputSpec(shape=[92, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_118a7b73b5b6444bc0207a52114ae696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bca3274da3af2c98c2af149f3c20cef4
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.uniform([92, 23], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a74448c45e53eee0a33665cf0b9a559(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 23], dtype='float32'),
                paddle.static.InputSpec(shape=[23, 92], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44281200be500f1e846e6764f7b31cd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a74448c45e53eee0a33665cf0b9a559
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.27309513092041, 5.833820343017578, 5.503009796142578, 5.4817891120910645, 5.425642013549805, 6.097849369049072, 5.734645843505859, 5.922000885009766, 5.353610038757324, 5.361513137817383, 4.979347229003906, 6.077544212341309, 5.27100133895874, 5.504151821136475, 5.766942977905273, 5.257235527038574, 5.357554912567139, 5.529504299163818, 5.592182159423828, 5.393463611602783, 5.493752479553223, 6.035822868347168, 5.610565185546875]], dtype='float32').reshape([1, 23]),
                paddle.uniform([23, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_58b0c58703f214c8839d4aa5857029cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbd9b411c1fe5c48704fb40e814a5df8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58b0c58703f214c8839d4aa5857029cf
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf7e9cb4c18985a9017932c4e925b000(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[54, 3, 64, 198], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0349da1765f58ca1004e804b7f5040b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf7e9cb4c18985a9017932c4e925b000
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c70a0cf98597c967a967d8a9dea66c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 3, 198, 198], dtype='float32'),
                paddle.static.InputSpec(shape=[54, 3, 198, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef02836eb71d045a23e527c6fc305116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c70a0cf98597c967a967d8a9dea66c5
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a15d33317bfdc48a5f37205e576dae9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2191c8ba6414522a23afee15b581c21d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a15d33317bfdc48a5f37205e576dae9
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f8e1b99154491d77e3a9fd668fdbf8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1960, 16, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ecee529981150ca83325b50a23f43c9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f8e1b99154491d77e3a9fd668fdbf8a
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9d647c6fe0b3646b612381eb51e0130(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1960, 16, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4568105aa2538d9152785fc4dd23091d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9d647c6fe0b3646b612381eb51e0130
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2218dd6b3d21675749b47a0e0ec5d883(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2048], dtype='float32'),
                paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44bf743d8eeaf9f0d6bb5893e68dd0ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2218dd6b3d21675749b47a0e0ec5d883
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ed11911d16ea35d7bf55f00e92731bc5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cff105b14938dc809ee065ed9d283d9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed11911d16ea35d7bf55f00e92731bc5
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_766518ba1fdff3b0d8f67093cc25b0c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f12a370e29a2f403252822c54f969dc
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_80872c8134cd319200014fca94b39922(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960], dtype='float32'),
                paddle.static.InputSpec(shape=[960, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb8256f8023e4048da4d0da39ae98b2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80872c8134cd319200014fca94b39922
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([960, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6324e384f896d8eafcdb6d6fe2b6f6c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 960], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90f722851e3a774319ada5bbcffb63ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6324e384f896d8eafcdb6d6fe2b6f6c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 960], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9401546718ea28e477eb061c6872ed19(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7add15ea159a98c85a9b9e818c803610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9401546718ea28e477eb061c6872ed19
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ebb508056ea58bd21af93a742dc1777(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1361905c3338dec8e6df3964fe25737(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ebb508056ea58bd21af93a742dc1777
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_76a0274a9913d0b73a94352c47dd8bcd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[512, 12544], dtype='float32'),
                paddle.static.InputSpec(shape=[12544, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bef2b7fd1ba0358625a0aa89053bb4e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76a0274a9913d0b73a94352c47dd8bcd
        def get_inputs(self):
            return [
                paddle.uniform([512, 12544], dtype='float32', min=0, max=0.5),
                paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_205ec42b3c95f5488005202d1843a3ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5156c712aae118bc15e8f8fd5209c830(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_205ec42b3c95f5488005202d1843a3ac
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_91a179c00586ea42a335eb916499aa61(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d33fb120b5d63f633a6b087956c540d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91a179c00586ea42a335eb916499aa61
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a8f1d1f5e07bbaf47cd41cb2905f072c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 84], dtype='float32'),
                paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0c76dd4dab7142d2756484773319dad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8f1d1f5e07bbaf47cd41cb2905f072c
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb1178f6e69ecb9790f34dd94f203b0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67d5bb8524e6ba72572e3e5489790f39
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f3ea12700b224cff233285f442b17de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41ddfa569cc29952d1ae78b76d48cbfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f3ea12700b224cff233285f442b17de
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ca7e819754d1c34230fb97c7e238dd2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbfb581dfe62be94733b2f1b391c59a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca7e819754d1c34230fb97c7e238dd2b
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8061e86ce43e12a56b9f53b715739136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07040b517a1ae60938156d75d10a0c1c
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf0b8de9e6806c6887052ab7f3aeb821(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d7ecd6a9603c4ce21466358c0a21f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf0b8de9e6806c6887052ab7f3aeb821
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_135af69e3475911f256f6cc2720520ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_045fd0bdbcb6c9fc5176d7c81d7b0725(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135af69e3475911f256f6cc2720520ca
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8118778a4a447e952f2476cf16bb43f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92b4f6b1b964afdb453546400133b953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8118778a4a447e952f2476cf16bb43f5
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ced0e2c357b25b20f266f123607f1e93(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 84], dtype='float32'),
                paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ca7548fccf0c0715e50ce0eeb36faab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ced0e2c357b25b20f266f123607f1e93
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2474ddf599b2baf170168162ca048dab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 2048], dtype='float32'),
                paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd91821c6327c83c5ed9869e545db583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2474ddf599b2baf170168162ca048dab
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92b4f6b1b964afdb453546400133b953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8118778a4a447e952f2476cf16bb43f5
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ca7548fccf0c0715e50ce0eeb36faab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ced0e2c357b25b20f266f123607f1e93
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a889bb677064c24888a414d98442e102(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db2499e31a2c2ff46f6ae4822eb02ac0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a889bb677064c24888a414d98442e102
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f149edf44b57c0e4f1664c4307600fbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_632025868c55048f38a46fccd437af11
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b38ae1adff67885d5b67393c6a2a816f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_442267cc882f83d06fe3b9610c5aeab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b38ae1adff67885d5b67393c6a2a816f
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48c66d7a3cbdb1edbe53fbcd9c0b87fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 4, 32, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3cf38cf34c0963fbd04f60c6183dd147(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48c66d7a3cbdb1edbe53fbcd9c0b87fa
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a2e836aad9d38ba97068b8ef1e9bd14(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 320, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 4, 320, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce790418fc8e3ef0f8305d8c3786b148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a2e836aad9d38ba97068b8ef1e9bd14
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d369f14e1e18afcf6448ecf001eec528(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a630bb4e0236660cc2856fa6cb35d1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d369f14e1e18afcf6448ecf001eec528
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_30f7c133a50939585765a884ee284e44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db57ad65edb02d9fbaae1234a7668b9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30f7c133a50939585765a884ee284e44
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dfbe5f3af54aab645626214dbc158681(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2277c49cf407edd71df133c6b0de7d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfbe5f3af54aab645626214dbc158681
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72183e5c2ca039d71d8c47892925fbc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_787c99fb19c95946e367ae8df64e2f62
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d8a75e0049615acf25c2ec7393df68c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29ba55c4d4b7055bbbf81caf927a2b8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d8a75e0049615acf25c2ec7393df68c
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2dd1c54e776bd9b4fac6fa7b4a02cf7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 12, 64, 577], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0ab120d6ed2bfb16a5cd18391ce2ea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2dd1c54e776bd9b4fac6fa7b4a02cf7
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 577], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_381b32a99a31b6ba37197cdc66d6461f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 577, 577], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 12, 577, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07c205c2cb0497f93bf227568b960e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_381b32a99a31b6ba37197cdc66d6461f
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f616f249665b615d1a24809a54da0d06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e9f65b388f8968c6773b6f01a28b1c4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f616f249665b615d1a24809a54da0d06
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_78f450eeb7181708e35112ec7e4bfbf6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6cb1ea0cf7ae5a66f8c580895c8b8f8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78f450eeb7181708e35112ec7e4bfbf6
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be000315c822a40c34d38a31e397e134(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2030f054ba8a76d85de335e842d1b474(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be000315c822a40c34d38a31e397e134
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_861415f8dcfa02f776f8694280f08555(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 197, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7359e1f187a9560c7f3f0c3e0f7891e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_861415f8dcfa02f776f8694280f08555
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a7781d5c3a73045ed81ec904c4b2a1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 197, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21798fe8edf42b7b3220bb9893d0ee8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a7781d5c3a73045ed81ec904c4b2a1b
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_88487874c226d058bf3819efa86b5dce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 872], dtype='float32'),
                paddle.static.InputSpec(shape=[872, 218], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1fcfa278527340a5d97873db13b4eb56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88487874c226d058bf3819efa86b5dce
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c138efa974dc9345470f8af58c5fa75e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 218], dtype='float32'),
                paddle.static.InputSpec(shape=[218, 872], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a5e423200aa6ab842fc11a12449c490(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c138efa974dc9345470f8af58c5fa75e
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83d5174cbbb07db08d46e46c23e43bad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d491283d35911575417543ce8789ec9d
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d63f7e82bb2ef9214b29dd18465ae7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ff7f1cd493d085e32d5033be66b44ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b9ee0f17d1fcfdfba3939ae6d666608d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c15b1e036155c3cbfb91414c77b61f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9ee0f17d1fcfdfba3939ae6d666608d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9798c41d7b549105a1b92ac30746a56a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 64, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0358d2e1224c48c6ceca850ded61b26d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9798c41d7b549105a1b92ac30746a56a
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b84b7cede4d9dee272a66eeeb061da35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 1024, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ca55119125d65a2745a4ceb72c40acf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b84b7cede4d9dee272a66eeeb061da35
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d63f7e82bb2ef9214b29dd18465ae7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ff7f1cd493d085e32d5033be66b44ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db2499e31a2c2ff46f6ae4822eb02ac0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a889bb677064c24888a414d98442e102
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f149edf44b57c0e4f1664c4307600fbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_632025868c55048f38a46fccd437af11
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f9aac4f7a135e9103a7ae943a871e65(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a62da6603ffaccdae97e6deb6b9602f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f9aac4f7a135e9103a7ae943a871e65
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f3b729119fd8dc06e858060265f270d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_26c0af891e4cbcc39032fab78553f4c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3b729119fd8dc06e858060265f270d5
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b5ac2b41fc5ccabd43dfc039223e40b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[390, 3136], dtype='float32'),
                paddle.static.InputSpec(shape=[3136, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_621973ad09a88bb9cea20657e8a506fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b5ac2b41fc5ccabd43dfc039223e40b
        def get_inputs(self):
            return [
                paddle.uniform([390, 3136], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_443367d6217482a4759c011a71df7035(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[390, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63d851e878f8356f18c3d7c8821798cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_443367d6217482a4759c011a71df7035
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_314b0517e96c9422b1ac1a5cabc08af1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f54d321dcea016421d9a33bd2456b99e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314b0517e96c9422b1ac1a5cabc08af1
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1c1c6b84c1b856654b6f673c7edc656a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 84], dtype='float32'),
                paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf60d6e39605d7931c28db8256090bd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c1c6b84c1b856654b6f673c7edc656a
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ff4af4822feb32562ee9cd1dffcea8a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60df4706e4d8d4d469c95aa8e2468965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff4af4822feb32562ee9cd1dffcea8a4
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9107be76759fcbd803bacfdbd5e2dc4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 2, 32, 640], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ec6878af4ff10f354158f7ac41fda17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9107be76759fcbd803bacfdbd5e2dc4b
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7f6120edc307e97b49f1640f6fb1ffe7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 640, 640], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 2, 640, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_544f5f52afec9d9c0474f38a76ee706b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f6120edc307e97b49f1640f6fb1ffe7
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_129aef1871b2f148a55037f4f2257e47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d24bc1d01c2456b8a8cbb4f66d24f4ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_129aef1871b2f148a55037f4f2257e47
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ab7d9b4a7e1be6cb2822011a8ddf9d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd70b31c9e8ef371c99d4f9075135ef3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ab7d9b4a7e1be6cb2822011a8ddf9d5
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d68ec7d2af7a6a64fc548e665b163dd3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_add47e4d077a0847b3d14c48c2d1a8e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d68ec7d2af7a6a64fc548e665b163dd3
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_976ec1cf0bb337971af065e802af27e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d4268efd254cee4ba0f1796a9260c62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_976ec1cf0bb337971af065e802af27e6
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9e64915163bf06400a364816f3ac5c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[86, 3, 64, 198], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92c1db52c36255abfa616e210ad4104c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9e64915163bf06400a364816f3ac5c5
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fec5ea523560e32beb0701941aa6d7f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 3, 198, 198], dtype='float32'),
                paddle.static.InputSpec(shape=[86, 3, 198, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3b7cab708c0ab9b8b19daa023b4fab70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fec5ea523560e32beb0701941aa6d7f
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_76d92f58ecbecc71519a21d3149687b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2ac9b28887e262ea71d0869a7204d85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76d92f58ecbecc71519a21d3149687b6
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a952c10d0f51b1bc3755f7f3916d96f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3b845abe3cb52e2b76c8dc500d3dfd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a952c10d0f51b1bc3755f7f3916d96f6
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2263dac67a6ac445a106868278979f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b474325fd7b61c4cc3aa6dedd346e3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc0cb0410841631a0a4403f89d014e36(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0220cb7ba3255ef485b8b736824c0148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc0cb0410841631a0a4403f89d014e36
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0220cb7ba3255ef485b8b736824c0148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc0cb0410841631a0a4403f89d014e36
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb7e1784937fd5f819c05848c1e15617(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e35dc8f6d9d9749f22d13b94233420a
        def get_inputs(self):
            return [
                paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9003a94099e638bbae30937ea5351f56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a4305a9635871917bc9ec504f508f75e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9003a94099e638bbae30937ea5351f56
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4305a9635871917bc9ec504f508f75e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9003a94099e638bbae30937ea5351f56
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7bc475fe1002b155394604404f019d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 2048], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17c6e9f8fead5673a6360d8dfd59cd49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7bc475fe1002b155394604404f019d3
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 2048], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1d69fd2503d963baaf52244360584f8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 169, 2048], dtype='float32'),
                paddle.static.InputSpec(shape=[2048, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eec0b0f3e49f2d55ccb6153528676c27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d69fd2503d963baaf52244360584f8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25fb9d2c37e308d50756373c5728a73a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f85e6c5e22a1c5fbc95feb4d88b2f20
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3555792507f6c708dc776761a6e901f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4312, 16, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c809ae953b0cc7ddb8277c0542c4da5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3555792507f6c708dc776761a6e901f7
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d1e8ce34ef5837ac72aa1a2de8903425(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4312, 16, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7907b78d6fb559c2e291cb06c0afa0fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1e8ce34ef5837ac72aa1a2de8903425
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d33fb120b5d63f633a6b087956c540d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91a179c00586ea42a335eb916499aa61
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0c76dd4dab7142d2756484773319dad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8f1d1f5e07bbaf47cd41cb2905f072c
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb1178f6e69ecb9790f34dd94f203b0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67d5bb8524e6ba72572e3e5489790f39
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3a7846ac7f60bc02082726cf167d91c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 2048], dtype='float32'),
                paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0dc99c4f004f45adf2ec808862977778(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a7846ac7f60bc02082726cf167d91c4
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5c0f5c58fc67af93e106f90841ccb91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90f60f0a56d773a21b1e19fd3c729cc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5c0f5c58fc67af93e106f90841ccb91
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aecfcfa8af96420c5c2f721ca0d478f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 9], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41cae9a25cd73d9210fba25887f3a490(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aecfcfa8af96420c5c2f721ca0d478f1
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac81d8643988d51173ac8a79d36ad3e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53fb1d8dee42d5d3273e2175f1885a70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac81d8643988d51173ac8a79d36ad3e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b99125a5efc34d797f001a2bdafa83b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bee32c439bd3096098284d1369010f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b99125a5efc34d797f001a2bdafa83b2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dacad921a58922ba15e26dfd59479515(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8, 32, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11048d9645bf108019481827a3da8218(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dacad921a58922ba15e26dfd59479515
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_088746bca59095378fcf199351863978(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8, 512, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_403608368fee84fea0703ef254405291(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_088746bca59095378fcf199351863978
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53fb1d8dee42d5d3273e2175f1885a70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac81d8643988d51173ac8a79d36ad3e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9567cf019544d0ee1cb5d6aad25b4f08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1280], dtype='float32'),
                paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e00dbb38823fb6dd1d17bf97034b9e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9567cf019544d0ee1cb5d6aad25b4f08
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c2aad62651e101766014b8125bec7ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_044fe339f1e862d4f353dea852b1d20f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c2aad62651e101766014b8125bec7ae
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b44e4b1ec0ee5fcf260032a30264329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e279bd219c149a8523b6137b56abacf
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10ff74d4cdfd0b9c3b5188fb89a7f631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a32489827fc616eb37c9af942cae24
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c6565e79ece1a1df6d2e76e6fb67efa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f528afedd4658a4504889b3f74223ff9
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b08e2fd293b49f5b9e6e3714bada9949(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7cf3a1855054150f51c85ab0accf41e3
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78e6cfcd4d29bc1c837e341e318c8a20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ef071c18706ab2be3f9c6c96756d18c
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87e005fe4b18ba744f3ef63bc19333cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_257a6be379031325eb646212524496ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87e005fe4b18ba744f3ef63bc19333cd
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c43626c3d4c74a23403629ad58d38d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_135936985f818323056facc9c7e940c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c43626c3d4c74a23403629ad58d38d6
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b379d886fc43212019497537ecb92b47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ead8a8ba59568248ac78854ba0c84c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b379d886fc43212019497537ecb92b47
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c558ec02b318572d9f79525ab7d6dd92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 336], dtype='float32'),
                paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c695e50b7256611dcb689b70ef9db16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c558ec02b318572d9f79525ab7d6dd92
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ab1563d6971b8a0a25275521ac3c59eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 84], dtype='float32'),
                paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b78dfaa09c458ff043a828e91e9befbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab1563d6971b8a0a25275521ac3c59eb
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f909b76832bb5d455d29e6062fdc29c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6c42be89d5c153f12d243c4ad73b299(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f909b76832bb5d455d29e6062fdc29c9
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_135df451f8f573d98a27af1c8cf169c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c985aee7a687692cafe48414bc503293(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_135df451f8f573d98a27af1c8cf169c9
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f54d321dcea016421d9a33bd2456b99e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_314b0517e96c9422b1ac1a5cabc08af1
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf60d6e39605d7931c28db8256090bd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c1c6b84c1b856654b6f673c7edc656a
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc88cfd6d8d2114e90c1c294a7a15a07(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 1536], dtype='float32'),
                paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14a9b070e6c3afacdf54889df0303407(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc88cfd6d8d2114e90c1c294a7a15a07
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6418e37341f2c9317bd6901f4b1fb392(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39c98936180a7111d700d1245d92381c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6418e37341f2c9317bd6901f4b1fb392
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ba0f3b97a8a59b485138510177d599e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9010f73c0286a6e7f4bc7817406b095b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ba0f3b97a8a59b485138510177d599e
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a530bf9e47a7b4e6493b65a43ec2fcd6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17ecbd9873e37b595b72672cd4ae1f6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a530bf9e47a7b4e6493b65a43ec2fcd6
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3a3590daa6e7e4c4264050199b076246(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01e0ff4838ca7ef25709e5eeff22a9ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a3590daa6e7e4c4264050199b076246
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0bccdd253ec51f306c74ab4980ea021(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 1536], dtype='float32'),
                paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37e5ff8826d80c90a7d0c4f8fb87238d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0bccdd253ec51f306c74ab4980ea021
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83d5174cbbb07db08d46e46c23e43bad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d491283d35911575417543ce8789ec9d
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44bf743d8eeaf9f0d6bb5893e68dd0ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2218dd6b3d21675749b47a0e0ec5d883
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_560669ce549b9b7c152e643c28e760e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0c6d3cb6819db3aecb04ea6e2df6937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560669ce549b9b7c152e643c28e760e2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_31528bb0c09066ddfe50f3b24af2a643(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6, 64, 1025], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8be3b14c71f7d67e9c11349b9f925f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31528bb0c09066ddfe50f3b24af2a643
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f520f0c11f8d34885399705e81a013c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1025, 1025], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6, 1025, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0993c661de1ea173cf5c46adea90638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f520f0c11f8d34885399705e81a013c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_13cc1a3d84ad9ed3737cc9f2e92e9013(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea6006b313292aa716ae4cf2108144d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13cc1a3d84ad9ed3737cc9f2e92e9013
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_65ef5357ce5074f23579e66774f56963(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_218fc98ddb1f0064f8c59a60b211b0b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65ef5357ce5074f23579e66774f56963
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce863bac51114723ca42b73f3f9a8ed4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 9], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a36a79a914148f42a256f91c832939a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce863bac51114723ca42b73f3f9a8ed4
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a1fea64274a20d366da2ae34fb4aabc0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd86b91ccff18dc16b26e3e39ed9d442(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1fea64274a20d366da2ae34fb4aabc0
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa218275c61399afa0aed219438fe085(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3bae2a84232e3a37198a5cf5b5fb78a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa218275c61399afa0aed219438fe085
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f714f222ccae55513b6720a747e287b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d182820710a5ff87b6e31ff44fc7ff43
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 150], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_37cd9e14ad47fa270e7965e52a2a5127(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23ac4f187be39bf44d27e7d6625297f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37cd9e14ad47fa270e7965e52a2a5127
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7090c31bdcee06ed88db0346f45c2d4e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dca5f6b33c284e7d0983ecfbd6ac7b7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7090c31bdcee06ed88db0346f45c2d4e
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aeca1751e0c154728d6ce9ae3414b676(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25695e97d1dc24df531c31b8360eec93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aeca1751e0c154728d6ce9ae3414b676
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e107c739feb53f22da704186e213199c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 672], dtype='float32'),
                paddle.static.InputSpec(shape=[672, 168], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d8acaa265c3c910b248e2cf87c2b74a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e107c739feb53f22da704186e213199c
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9c9af2b76b5b98c938477a32d89e82af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 168], dtype='float32'),
                paddle.static.InputSpec(shape=[168, 672], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a54c5d7e40d36f5f67cd5508eabdfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c9af2b76b5b98c938477a32d89e82af
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae1dc3ffc27f3c09cec0fa2459fee654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b03f2a8c6217147961f891278adb558
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2a5d273e00ea4700175b8a70b97d8ca2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_076393a497b2794b605df3b43d94f9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a5d273e00ea4700175b8a70b97d8ca2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e3b413610c8577ab3ca2378c6ae0af2c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 32, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1f0bf248e2d506bd4e68050ffa12f6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3b413610c8577ab3ca2378c6ae0af2c
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f66a0d3c4a212b27088b7b58ea12aedf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 512, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0df696acdabb469bf94bfccd09a5eb07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f66a0d3c4a212b27088b7b58ea12aedf
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae1dc3ffc27f3c09cec0fa2459fee654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b03f2a8c6217147961f891278adb558
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b9aeb40e3a47a7c28e6ddd5a975ea25f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_924edc65da25e27bfed1e30092ed9b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9aeb40e3a47a7c28e6ddd5a975ea25f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_05130ef00696f43de3adcb7192232203(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09cdd62c7f46194f42be01bdd6185059(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05130ef00696f43de3adcb7192232203
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0d6f78927f7bc0a93b5c361f341be5d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8, 32, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b06a8eac6635d898513e32f88c3dabc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d6f78927f7bc0a93b5c361f341be5d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_47d9e3a5fc9507e20d756c6378edcb4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8, 1024, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b14f1a499a4c4f57d794efe20909aee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47d9e3a5fc9507e20d756c6378edcb4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_924edc65da25e27bfed1e30092ed9b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9aeb40e3a47a7c28e6ddd5a975ea25f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c695e50b7256611dcb689b70ef9db16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c558ec02b318572d9f79525ab7d6dd92
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b78dfaa09c458ff043a828e91e9befbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab1563d6971b8a0a25275521ac3c59eb
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ceb9831b1f870488f77eb92c04013cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c93c09ee72217202b86bd3e8c52a6d79
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37d5ed89d17fec89865171b50b0c5287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcf38b0c107da901d39def7e75bab947
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c098ac0f05d4c9c502c8746b0f6d19bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2048], dtype='float32'),
                paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_605925b4b923db30da3f1dd70a72d1f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c098ac0f05d4c9c502c8746b0f6d19bd
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44bf743d8eeaf9f0d6bb5893e68dd0ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2218dd6b3d21675749b47a0e0ec5d883
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9511639f2d25e9e0dd628bffa2d70538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14ec9053ff1c5849d3e346a2cf328895
        def get_inputs(self):
            return [
                paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25fb9d2c37e308d50756373c5728a73a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f85e6c5e22a1c5fbc95feb4d88b2f20
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_560ea7c5e96668dbcf6d4928dfb5b1c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd550c425031b4ce932a9902ec714f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560ea7c5e96668dbcf6d4928dfb5b1c4
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f6b9c50707fda4b3de2f40796d40452a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[54, 3, 64, 197], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4217b751b521974c38bffbf2114192f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6b9c50707fda4b3de2f40796d40452a
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0cebc1f6f0ddd0ec01b2586472a47ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 3, 197, 197], dtype='float32'),
                paddle.static.InputSpec(shape=[54, 3, 197, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c32c10cf45ebee3df0331dd19910fad8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0cebc1f6f0ddd0ec01b2586472a47ae
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49a0906586d36421970de9cda22ea590(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02018c381820e5303fb5c2d09c57dbc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49a0906586d36421970de9cda22ea590
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81b27a43b38fbe2b3458f69c5967c5d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87d1efc1c6d891ff3301363c838acb04
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ee660226e95406f9614c1e9e5c91c7c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4351b18205f926b065e6243635b80e63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee660226e95406f9614c1e9e5c91c7c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19b3381943859b850e014cc74c1759b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 32, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60c143845153bf0d34c453b51590c2ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19b3381943859b850e014cc74c1759b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3579762bedee23efc2a95a78408a9a2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 1024, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0cb4951a46b3141e39ad5e22cbf67a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3579762bedee23efc2a95a78408a9a2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81b27a43b38fbe2b3458f69c5967c5d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87d1efc1c6d891ff3301363c838acb04
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_20d82a1b018b74f31a2d2dfd4c0144d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e34736b40660f026475e3bcd9d4bfba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20d82a1b018b74f31a2d2dfd4c0144d7
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72183e5c2ca039d71d8c47892925fbc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_787c99fb19c95946e367ae8df64e2f62
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3b845abe3cb52e2b76c8dc500d3dfd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a952c10d0f51b1bc3755f7f3916d96f6
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2263dac67a6ac445a106868278979f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b474325fd7b61c4cc3aa6dedd346e3
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dcdda9479ef6a9177b3ff5a3fc4e9694(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 40, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 6625], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7598f5448931ea8044ecaf6d88f3107f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcdda9479ef6a9177b3ff5a3fc4e9694
        def get_inputs(self):
            return [
                paddle.uniform([10, 40, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 6625], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_442267cc882f83d06fe3b9610c5aeab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b38ae1adff67885d5b67393c6a2a816f
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cf38cf34c0963fbd04f60c6183dd147(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48c66d7a3cbdb1edbe53fbcd9c0b87fa
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce790418fc8e3ef0f8305d8c3786b148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a2e836aad9d38ba97068b8ef1e9bd14
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a630bb4e0236660cc2856fa6cb35d1ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d369f14e1e18afcf6448ecf001eec528
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07ca628ef4e0b507fdc994ac65094013(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3170531079b29050f8dee339fbfc17b3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bbbe67342a997f9b0993be350a9350e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57133ad7def1286178ab36cc1c18dd94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbe67342a997f9b0993be350a9350e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e1f7ac209e5d38389ca9d89b3d827e40(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 64, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49ff67bb7950dd86adce8636716854d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1f7ac209e5d38389ca9d89b3d827e40
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1fe6c759d2489cf018b7c582c990b52e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 512, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b34a7bd595696f0437f26223086fd77a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fe6c759d2489cf018b7c582c990b52e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07ca628ef4e0b507fdc994ac65094013(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3170531079b29050f8dee339fbfc17b3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd86b91ccff18dc16b26e3e39ed9d442(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1fea64274a20d366da2ae34fb4aabc0
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bae2a84232e3a37198a5cf5b5fb78a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa218275c61399afa0aed219438fe085
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e1fb213662326852a36829499030e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b59525f41035eae3778e6f38efc550ef
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 150], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_33a3d717bb269257e9bdf317b89d632b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ec48fb8b0a971eff92ee6d68615e4ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33a3d717bb269257e9bdf317b89d632b
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e21d4f69d345cd8ca777d2642270e76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 2, 32, 200], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7b3b292087cbb05e9848f2083acfa19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e21d4f69d345cd8ca777d2642270e76
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fce66c0c46390db85ec726ee9d067689(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 200, 200], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 2, 200, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8125b142fe46e4e7b09de1af74f23b78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fce66c0c46390db85ec726ee9d067689
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_52f8c0b2feb9d2496b74deec95a9075d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_428270d8fe80d988bab923c849a7ad7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52f8c0b2feb9d2496b74deec95a9075d
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_859281ec58572c869925287e0f98cbf2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ebe8b80a4ff645790fe95e77137c3f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_859281ec58572c869925287e0f98cbf2
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1971faaf1a49f4dffd21900af110f8e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 197, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b223178a382db1971fe2c8362400177e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1971faaf1a49f4dffd21900af110f8e4
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9d6cf6755ef120d52c674f542a2fdf72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 197, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82a598403aa65d2cd020fd9344018686(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d6cf6755ef120d52c674f542a2fdf72
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bea6c9d8a006391d1a636e0ca43d709c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f71edb2229db75839c660170be6769fb
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c855924d24791950e43a4f2cf32e151(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c63e8c715f82ba7ee01ca9bbeed48072
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4206ba0fa446df072fd8df7f372f727f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fa9288592453ba2f4ef5edb593e983b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4206ba0fa446df072fd8df7f372f727f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_392732b4cb343862bc86cef1bde820e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 64, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d93083c513596aa390f17ebd0ee6940(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392732b4cb343862bc86cef1bde820e3
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86115ed7e0155bb070eb738c6032ba0f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 512, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cf044909cb1c432cc7630a20a840cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86115ed7e0155bb070eb738c6032ba0f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c855924d24791950e43a4f2cf32e151(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c63e8c715f82ba7ee01ca9bbeed48072
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3b446ad89af37f9858323e26b21d990a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 704], dtype='float32'),
                paddle.static.InputSpec(shape=[704, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_972974fdd9a5a0aa03e2c3356b83c8f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b446ad89af37f9858323e26b21d990a
        def get_inputs(self):
            return [
                paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe15ba72573c8c007f2635aa1ede0813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f08af081a1da3c181690cee7d4f746aa
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7e5d8c20d209ec6ca0f454760ce14af7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 640], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b96bbe6c095929379c7b41d645941e9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e5d8c20d209ec6ca0f454760ce14af7
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e36e3fcacf849f057a3947ef786a7fe4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 64, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cca1608d67c87776c1fb9b03daea9a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e36e3fcacf849f057a3947ef786a7fe4
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_36d36aca674b0c6c2b03d120d9d4b729(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 512, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff4515ba3a9db5a4e2b571edb1bed5c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36d36aca674b0c6c2b03d120d9d4b729
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe15ba72573c8c007f2635aa1ede0813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f08af081a1da3c181690cee7d4f746aa
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ad97b6da8c496136e999c0994e4f993c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1248], dtype='float32'),
                paddle.static.InputSpec(shape=[1248, 312], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf2bad3b7d580fc575f2c5566f9fe7bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad97b6da8c496136e999c0994e4f993c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.uniform([1248, 312], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2b56db27be30606b8d26f9ce8659689(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 312], dtype='float32'),
                paddle.static.InputSpec(shape=[312, 1248], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bee78f671c90ec058f72c3d6bcca5393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2b56db27be30606b8d26f9ce8659689
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
                paddle.uniform([312, 1248], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df5f3d6de4044afe5c92a037215ee7eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94346b163b8ac10dc6ed4be4548f8486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df5f3d6de4044afe5c92a037215ee7eb
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3654bf205f70fee81521980cc046a3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0505d43d7d64c8d9536e30c61b8ad15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3654bf205f70fee81521980cc046a3e
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_803b212d473d07a75a74cac04e40b62a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc68e58172b7c1a375d31dd7598e295a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_803b212d473d07a75a74cac04e40b62a
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fffe9e8282914cf55ef15aeb79cb9175(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_acca6210b2f6eacd4df656df0fba55f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fffe9e8282914cf55ef15aeb79cb9175
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5b32f7f63787a9874c5cf5e7e922d61e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 9], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcf688ab511cfeb2f53053dd6059cdad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b32f7f63787a9874c5cf5e7e922d61e
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60332e84d019192def985e3d47f7638b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_490e1a283278b7099ac0ad4bfd241604
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9d0973c3003a0de6df2c019b81a0d82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed73457c1949b89b342392a288def26a
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60332e84d019192def985e3d47f7638b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_490e1a283278b7099ac0ad4bfd241604
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_194159e71115d365da9170fa03e7e1a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_550e90e711bfb89ed7327f2299bdae83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194159e71115d365da9170fa03e7e1a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf13183741806396503e8d313087ec1a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 12, 64, 1025], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b4ff9febfd57dbc3129100c5ee8f7aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf13183741806396503e8d313087ec1a
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a3fbd51abc0673d81bb77717f66f3af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 1025, 1025], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 12, 1025, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e764625edd408dd98a5d7089418346d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a3fbd51abc0673d81bb77717f66f3af
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aff2bfed4aefbb74f24dc290be994810(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4502d878a7f905ce4f539256fac1e227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aff2bfed4aefbb74f24dc290be994810
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9d0973c3003a0de6df2c019b81a0d82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed73457c1949b89b342392a288def26a
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_accdeab446e8a6e8b1a0fb148ee192d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
                paddle.static.InputSpec(shape=[156, 39], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fda710a109a62573de2d17136a7c9e62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_accdeab446e8a6e8b1a0fb148ee192d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 39], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f32911c370e9a3ab8d8347b1d28d96bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 39], dtype='float32'),
                paddle.static.InputSpec(shape=[39, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6536eb93681208fb036049473bff2973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f32911c370e9a3ab8d8347b1d28d96bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
                paddle.uniform([39, 156], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_304385c5d3e20371ba5ddf7d908fabb6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2fea385ec08510f458e24e2b3d9cbea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304385c5d3e20371ba5ddf7d908fabb6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_33c0083ee3e7c3d29df5db185b224b0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5be8a5c63641da71cfc0ae761e7dd954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33c0083ee3e7c3d29df5db185b224b0b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_804cbfd4e199056615fc3256857751cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8, 64, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7dc43348ac29cce13682f0d59493319e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_804cbfd4e199056615fc3256857751cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_00a51c1c7300a0d36059e07c9a86d58a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 1024, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8, 1024, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44d5f39e7bf9e28a72b14425ce089e35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a51c1c7300a0d36059e07c9a86d58a
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fea385ec08510f458e24e2b3d9cbea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_304385c5d3e20371ba5ddf7d908fabb6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fcfa278527340a5d97873db13b4eb56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88487874c226d058bf3819efa86b5dce
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a5e423200aa6ab842fc11a12449c490(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c138efa974dc9345470f8af58c5fa75e
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c6a98a47382f3f693eacb6f0932fb3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f1211eb27f7c5e640b1825307ca6765
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bea6c9d8a006391d1a636e0ca43d709c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f71edb2229db75839c660170be6769fb
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_42e00d364ef2a224a9614bc37efaf831(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b6c01e3d092d7679e85b2eb1d19486a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42e00d364ef2a224a9614bc37efaf831
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_46f225a1db1469327cdf9798e848e4cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a59f0b8117cb3685f460ed325b1b3d7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46f225a1db1469327cdf9798e848e4cd
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a4fa60fe40c74dc4a02317c98153241(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5a9c75e87a7db2393ad6ed51a4338a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4fa60fe40c74dc4a02317c98153241
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3bf69b2c95c2165f0d18d58265b6972c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 480], dtype='float32'),
                paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57579f4ceaa0884fc4000d6be93cbe97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bf69b2c95c2165f0d18d58265b6972c
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34b3ea8e8acb6c1243cb8ddcc67b4a8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[145, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c104be3720bec6758bc900357c872048(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b3ea8e8acb6c1243cb8ddcc67b4a8a
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a62da6603ffaccdae97e6deb6b9602f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f9aac4f7a135e9103a7ae943a871e65
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26c0af891e4cbcc39032fab78553f4c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3b729119fd8dc06e858060265f270d5
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08e4a82184a7effe2addafaf222c0c04(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 25, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 37], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_716666a9ca57c5365f9d2ebfccabf1d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08e4a82184a7effe2addafaf222c0c04
        def get_inputs(self):
            return [
                paddle.uniform([10, 25, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 37], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3a9d3fe71f63d7a9e63121e920cca94(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39d8a6ea16727e70ce8b0f7a16e42e29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3a9d3fe71f63d7a9e63121e920cca94
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_395efa1b1c75e6301444039194f49862(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[171, 9], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f73822819c6888b1643a1ef59af075c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_395efa1b1c75e6301444039194f49862
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8eee90909085f3312931f9a0a22b7f72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84807f7ec68d68ff34f2f905c2cc95a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8eee90909085f3312931f9a0a22b7f72
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3bf2bef3686484c3b516f6fd055b9d59(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[30, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_741c1b993d035bdefcfa66ee5be65f4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bf2bef3686484c3b516f6fd055b9d59
        def get_inputs(self):
            return [
                paddle.to_tensor([[8.870004653930664, 9.070971488952637, 8.6998291015625, 8.771418571472168, 8.77302074432373, 8.278258323669434, 9.320741653442383, 8.51421070098877, 8.791091918945312, 8.457852363586426, 9.51916790008545, 8.142207145690918, 8.19686508178711, 8.17872428894043, 8.666762351989746, 8.191591262817383, 9.497175216674805, 8.180706024169922, 8.71663761138916, 9.759553909301758, 8.672922134399414, 8.273152351379395, 8.350447654724121, 9.068215370178223, 8.876349449157715, 8.625724792480469, 9.934979438781738, 8.560579299926758, 8.065917015075684, 9.131192207336426]], dtype='float32').reshape([1, 30]),
                paddle.uniform([30, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f8090631f1b66ca29f2043b79956053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4166d6a0c790e921533ec9c342f3ca69
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_92adacd32c68a9b5414b591a054aa98e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 640], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9828bcaf5d2badf5f99402b2f6491de3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92adacd32c68a9b5414b591a054aa98e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4cd0b51ef98ad06419eafbe59d817f56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 64, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc9535f880f89d6768cf1c013ee2c7b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cd0b51ef98ad06419eafbe59d817f56
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5519a334b2aa80ac7579ee694234cfcf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 1024, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15a6f6258a39df442818210757495c79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5519a334b2aa80ac7579ee694234cfcf
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f8090631f1b66ca29f2043b79956053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4166d6a0c790e921533ec9c342f3ca69
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bf7ab668f482a7f1427453fa04985ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7718206a77b312929288ac75bb98c50
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f1a830baeb9dd85e8c7b7128a85d08e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d95ecd97b64fe093ac77a03e3c54e37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1a830baeb9dd85e8c7b7128a85d08e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e40182b16f973607281cac3939ff6b52(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 32, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f487cf544c37b2f32e79763dc501d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e40182b16f973607281cac3939ff6b52
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_99af691f7c886761847d8116b9741b7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 1024, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7d0076d56b9044202296fd065bd7bc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99af691f7c886761847d8116b9741b7e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bf7ab668f482a7f1427453fa04985ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7718206a77b312929288ac75bb98c50
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8acaa265c3c910b248e2cf87c2b74a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e107c739feb53f22da704186e213199c
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a54c5d7e40d36f5f67cd5508eabdfe3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c9af2b76b5b98c938477a32d89e82af
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbfb581dfe62be94733b2f1b391c59a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca7e819754d1c34230fb97c7e238dd2b
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8061e86ce43e12a56b9f53b715739136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07040b517a1ae60938156d75d10a0c1c
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_412f27ad5080759271753ad203acb002(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29558b18a35d37cd458fd366fd1fc4e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_412f27ad5080759271753ad203acb002
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd64c3393faed44a03923e401d4ead24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_606cdc9c26a2e0f199eb8532508bd1a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd64c3393faed44a03923e401d4ead24
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6049149954bda8b9ceded9c8c99f1595(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8, 64, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1cb736d26b1440380ad4b56882cdbd93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6049149954bda8b9ceded9c8c99f1595
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a485ce8fd7651836e2938ebfb0a967f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 8, 512, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_373a5b346ea6f79f1637e26721e7a313(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a485ce8fd7651836e2938ebfb0a967f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29558b18a35d37cd458fd366fd1fc4e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_412f27ad5080759271753ad203acb002
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a2d20970fa1ebaf1737dffe2469a259c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb07ff03476adf0ebb8e7c0b4e5161e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2d20970fa1ebaf1737dffe2469a259c
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f354446fd793d5f25fb8c25357881a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b42b043d8d5aa781aaafab28a5900e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f354446fd793d5f25fb8c25357881a4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cff105b14938dc809ee065ed9d283d9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed11911d16ea35d7bf55f00e92731bc5
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_766518ba1fdff3b0d8f67093cc25b0c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f12a370e29a2f403252822c54f969dc
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_044fe339f1e862d4f353dea852b1d20f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c2aad62651e101766014b8125bec7ae
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b44e4b1ec0ee5fcf260032a30264329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e279bd219c149a8523b6137b56abacf
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_68368ec2939e033ab92968aec45293a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42c11bf5366575730caefea30f2b29d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68368ec2939e033ab92968aec45293a9
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef069fb472fb3d1e0c0dccdca21d6b78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cc3416bc95835ac2c7ec5c3525b7b71
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8a43f8c9c56f60c3cd8f9643a420c4a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a291c7335547678a791d4681589086fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a43f8c9c56f60c3cd8f9643a420c4a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_425116e4d40c6b40ad9f3fd79c079d26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 32, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ab98d7961d39389dadca9f9ecea4228(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_425116e4d40c6b40ad9f3fd79c079d26
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_060865cfa99714ac5f4c7cd38ee7d27e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 1024, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36e4e0682f8affe49c8a519acc782806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_060865cfa99714ac5f4c7cd38ee7d27e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef069fb472fb3d1e0c0dccdca21d6b78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3cc3416bc95835ac2c7ec5c3525b7b71
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c28b157225c1cbab737c5fbf55604733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9d6122628f26554e02e86885f67773
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57133ad7def1286178ab36cc1c18dd94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbe67342a997f9b0993be350a9350e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d732f96f974a5d852a2bddfe047a700(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 32, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a28305e85460f583cf0768bf387508b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d732f96f974a5d852a2bddfe047a700
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_12dc6fd440b34b46a13883c8c15268fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 512, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2384dbbcba4196cf00f96e0ab6a8fa74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12dc6fd440b34b46a13883c8c15268fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c28b157225c1cbab737c5fbf55604733(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9d6122628f26554e02e86885f67773
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a83258e77fcada1a2a9de2f437fd4109(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38bce2913a42c6d422c90bb8150b15f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a83258e77fcada1a2a9de2f437fd4109
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1f48f6ee81861b881ec2893122ffa370(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[86, 3, 64, 197], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_410fd3da6c5e29a4edc4028e5f5dfcaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f48f6ee81861b881ec2893122ffa370
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf5a9ecf35fb08048f30124562393532(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 3, 197, 197], dtype='float32'),
                paddle.static.InputSpec(shape=[86, 3, 197, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20d11c96c768ef7e59391615b0e57863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf5a9ecf35fb08048f30124562393532
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f732bbe1c7ce680280840c856f39250f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b692a3a295fcd81a5800ddeca28c8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f732bbe1c7ce680280840c856f39250f
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a4e321986a9aef2baa7fcc63dcc26d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45292ea15aec1645c0b03a819c1c83f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74e37fb924479a88daedcead133390bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_168716c3ca289a12d51bcb30fd409f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74e37fb924479a88daedcead133390bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_790fb958205a146283d71b3fc25f630b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 32, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a116be01e4161dfda16203ccf07ce3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_790fb958205a146283d71b3fc25f630b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1d050de0453c035cc8ce2d2dd1e3c5de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 512, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b6b1e4966459795f0bfd3f71c94413b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d050de0453c035cc8ce2d2dd1e3c5de
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a4e321986a9aef2baa7fcc63dcc26d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45292ea15aec1645c0b03a819c1c83f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa25ee820fac3da61176af6db85d26b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_79fa9fa553ffa2b6735690bbf404747d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa25ee820fac3da61176af6db85d26b6
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9495b6124cbb606ed1635e8cb3ffc46d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 8, 32, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cc176927adf3a0142cf51be9748d227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9495b6124cbb606ed1635e8cb3ffc46d
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98fa096df172890ec4f60d686489b83d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 160, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 8, 160, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a48563df38b52a0a6992c72b7cf6ef60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98fa096df172890ec4f60d686489b83d
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_22f5adf20b825165ddf68101f5bdaa44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9882dbfc5899cfaec6d2c58f4cdef13b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22f5adf20b825165ddf68101f5bdaa44
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_910513c63e2c60fb5ca1d1d581ff4172(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6fcb403386ff9bf3a419c7f02c0d4905(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_910513c63e2c60fb5ca1d1d581ff4172
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e6f84770c4d462ab7c6e086a3fba548c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6, 64, 1174], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c9c37ea7703eae12f913c2dbec9f8fc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f84770c4d462ab7c6e086a3fba548c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_75c41783a4220ca44efe5f67b7d9ec66(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1174, 1174], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 6, 1174, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e5387f4d92900b24deec8a76df9a984(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75c41783a4220ca44efe5f67b7d9ec66
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_830059d93763ba1e1641a359633037d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4bea3e682a108fe83d7fad503f787e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_830059d93763ba1e1641a359633037d0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d0068aa6162e9e17ffc1d6d8b2097f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1280], dtype='float32'),
                paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eb8b7b6413b9215cbead383be6412190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d0068aa6162e9e17ffc1d6d8b2097f2
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ba2bb13929c969aab431c692f186dc0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 704], dtype='float32'),
                paddle.static.InputSpec(shape=[704, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3345afbe7006243819115e271ffbe9e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ba2bb13929c969aab431c692f186dc0
        def get_inputs(self):
            return [
                paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c60f38b5804e9e0c9cb9dea476cfcb0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9557c5304535e44b8ee7a6b0f609961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c60f38b5804e9e0c9cb9dea476cfcb0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6bce394d0d8c940a5538da19d8a15ed5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 12, 64, 1174], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_074e6c889b177a1449802a88c8429336(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bce394d0d8c940a5538da19d8a15ed5
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_13fa8ddf353255bd5385474b64eee0a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12, 1174, 1174], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 12, 1174, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_16c31cf577fa05fdec4518c391c0a8ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13fa8ddf353255bd5385474b64eee0a4
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2879659353517f5a9e36a4edc7acb5b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_510d337de57781b274ee777c0bbe8e63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2879659353517f5a9e36a4edc7acb5b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d805a5b49d4493986dedde245077e4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9c51780aaf8bcae65efc3352bff8129
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a291c7335547678a791d4681589086fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a43f8c9c56f60c3cd8f9643a420c4a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_328f18e6d7f75d69ec9024f5db7ed777(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 64, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba16473959b8f16385b485b1fc6afcef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_328f18e6d7f75d69ec9024f5db7ed777
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_acbeb86692579d1193f5dc1c0176e1d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 1024, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5676c39dc945d0ad2ea2770b46407a86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acbeb86692579d1193f5dc1c0176e1d3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d805a5b49d4493986dedde245077e4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9c51780aaf8bcae65efc3352bff8129
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_66dad7e428dd0daae0bdfc3922007ca3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 624], dtype='float32'),
                paddle.static.InputSpec(shape=[624, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b11895002c3568c759f6131fd323e484(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66dad7e428dd0daae0bdfc3922007ca3
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.uniform([624, 156], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be212f29f37e636dd1dcbe3d3e212ce1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 156], dtype='float32'),
                paddle.static.InputSpec(shape=[156, 624], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8bc73546afdc8fbb61876a7fff7c8c56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be212f29f37e636dd1dcbe3d3e212ce1
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 624], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c1e61c11e021024b0d69f13ca6be3588(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e011130543ca5d967aa467272cde6621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1e61c11e021024b0d69f13ca6be3588
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_21a5b49540db5031ba6d856a13c9aa2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 8, 32, 50], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8457474d1df3ef1c15332f1e6e6c755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21a5b49540db5031ba6d856a13c9aa2b
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3f0ce930391c1f22da487d5d868174a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 50, 50], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 8, 50, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ce33216e6b321ae22054a7c7e8dbe27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3f0ce930391c1f22da487d5d868174a
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_035fbe1af0d566ef9bea15681b25d180(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0544f76399a965415a02e6d0b0dc0275(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_035fbe1af0d566ef9bea15681b25d180
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_777fecd4813047696fa89e251d1588fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a5301975d36e596755c8054c1f6a79dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_777fecd4813047696fa89e251d1588fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
                paddle.uniform([91], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f385054aac7d6b38c59dfc857791c857(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a6d1818f673c290d20f9b1b8f5fa478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_677996df9338359e68ede0e1f1df7b48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b518c7cdf6bb831db3801d6afdc71dfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1f39b376e6498b4ec435cbf5a7a66740(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4d2d0e4afae900f823eaaf3aec11554(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.626307010650635, 4.391082763671875, 4.806268215179443, 4.401785850524902, 4.369851112365723, 4.688978672027588, 4.786715984344482, 4.2709174156188965, 4.522144794464111, 4.13695764541626, 5.095004558563232, 4.644773006439209, 3.8188412189483643, 4.205009937286377, 4.518152236938477, 4.431371212005615, 4.864565849304199, 3.6953768730163574]], dtype='float32').reshape([1, 18]),
                paddle.uniform([18, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d952f5b95ced5f544dbba258ee4006e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73419629fa67b2e20e8e9f48b90ceda2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0576dc28522e90776dcfd03bcf2a51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce71befa51ab01917cb1741604182423(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a02800814414fa6d3d0fa738a832483c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.uniform([92, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4e79f137c15c7d11fe29d2a6c70e55b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[5.27309513092041, 5.833820343017578, 5.503009796142578, 5.4817891120910645, 5.425642013549805, 6.097849369049072, 5.734645843505859, 5.922000885009766, 5.353610038757324, 5.361513137817383, 4.979347229003906, 6.077544212341309, 5.27100133895874, 5.504151821136475, 5.766942977905273, 5.257235527038574, 5.357554912567139, 5.529504299163818, 5.592182159423828, 5.393463611602783, 5.493752479553223, 6.035822868347168, 5.610565185546875]], dtype='float32').reshape([1, 23]),
                paddle.uniform([23, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e194ddcace968be8ad2c0eadcced5fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c48961637e9f7cc0fac084250b84100d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af1766c51df13a0c7ea51037e13276b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a8654affe429f09cd4c70880b451b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_475b204d2ca6f65d71865511ea1d4e6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_338a48534018762a5d297731aa93f961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddaaf6d007f863b9c14c6280c1dc1a7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e51f55740fd190c47d8a51b2ca8d597(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f2728c9522bd461b430038f5456f104(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5291292b678fd2a2fdfd95078bb13c62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([960, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07dbd606f67cbbf340e86467ab61648f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bae593c0d0dfdb3df6df270b55f10a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e414cadfcf9ab90292b2a13276c464d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_127825fd94272645283d7aa5a634073f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([512, 12544], dtype='float32', min=0, max=0.5),
                paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b97314c094f09d28ef22f110b8e0dee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c10351b6671d9ce1f1ec8bff706a0a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5c29081fa30580d2e287248ca0a00c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37900175a18e469004fa58e07de71d45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_538c13f1541406032495df7e152d9850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0772e7ad90e364b830cc09c0045e5001(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9398fecb73cf9b706d832802793019c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb18ee0b86b06bbc5ec68fc35906f0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_408d12102d7847220db205c1ec2f1cd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3aaafff51cdc70567df55f7b4d9904f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c990ae277d0ce9e64b5727cef5c62c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e1ca59792b624ba40ae3f5ce695e1e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3aaafff51cdc70567df55f7b4d9904f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c990ae277d0ce9e64b5727cef5c62c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bb387eb8bea6843d631a5c5ed1dc05b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df133c44138585f09504c9cbe8b4360b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0e2e350ee5e2d0503c8a3ad12c5517c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f68525246a2dc8f705f887c3fc02f5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1708a94c5fae8d1dde586aee22b1e8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1deb090241a0d1f23be31c92c50c1214(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99f28843e60eeab9cac97637abe2744c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57798dd7f531a9ef3325d1c8798a5b41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67511dd00aabb1a4d5ee8810d62acd60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_252cc876e993c4469fcc95eb3a3bda6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7d9e75cd43d8aa0c29525cb440c802a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 577], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ea42b71adeca5204e3a11db39895b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e1e4e51cb5eb4776852b81446138a2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a47db5c125b207dce0f3cc9300d1c72b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a5f537185ebba051e6abe163bef0d36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6343c7b53e592e0f4c3392cdcb81d6b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce410565bf8022aef6ac72b12bdb1c6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33e15a8a206d2b02c9a72552c4eee186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dc1ec4da69ab9134e96843c3d4a3f35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5472f6622444f324db0769ba750d9be1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_baf115c4266ddae0eef8e4d3187ce3eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_103872c6cab321fc9dd14fa04c3f03f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7623a21057c556c6db9c5df4819bef79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5529e7765b1f75c78587d1f41cd740fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_baf115c4266ddae0eef8e4d3187ce3eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bb387eb8bea6843d631a5c5ed1dc05b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df133c44138585f09504c9cbe8b4360b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_548b84edf6d59eab422ac8f099676566(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83f06b0b73a58fb555cabde5e1289b1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e01c4c291fb072486b71c7ba0a608770(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([390, 3136], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f86ed42e26644aeb7ac8cdf563eb4c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ea6825e67c5be8c9da051532d99450c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_590a0d70e3940e03c96887d466df5b31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14fc8901f0186e5da8cd4136035f25e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_145c28d3c040b6fed72dbf4ab6962a46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5305180e1414ffbe5e375245f83d172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_512db71466ec52ba3c951c8c139d5929(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5f7e7c5179d2ffd71e43a9691ba16c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_371bd50a58cbecbaef91a798a5bd9f1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_910847ce5a421db3b35515fa6a56ca3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_162622aa342f3c9f98870bec03266170(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63a4c27f3352702f7090ff8433de7117(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_738d5a237a3d155f1650e106f562563c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_633cf9a8bfd6bbe35782a786a89b73f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bd8dc09981592fa276d775661e90aa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_695bd43e141b05cae4dbd9e4bad375d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_695bd43e141b05cae4dbd9e4bad375d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d1a9040eed9eec1d839cca459f7f251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4cd5d6849abc7443575e79999364383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4cd5d6849abc7443575e79999364383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e902cb0e99edd88658b27f87520b2fd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1287f28d82d55a567b28b5ec0debf816(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f415271ad99bb696c326169d58e4664(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e6ad7e6229a3d44e6fa268bb9b769e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fd724a047e9a3ff0b0b74c5605ecefe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c10351b6671d9ce1f1ec8bff706a0a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5c29081fa30580d2e287248ca0a00c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37900175a18e469004fa58e07de71d45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f93b4ded93f28233f58f868de07e20f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63920cb9c66e25dd054d566fb80c9ab5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2aa5ceb47467b1b20bdf5434c80d7068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acf1c1f2d83a042693f7eb5aae1ffbfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_465b8223058db87dcba112a476a63494(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b38eb8f501d1b223a01f3b792b31e944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b76a009a9fadd824cf613533fb7b30c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acf1c1f2d83a042693f7eb5aae1ffbfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3ff51a9d95cc114e4c75247935c12ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_010d25ae262d6943f3dc15dc1c0a6b51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12af8f64e4d8d3b67d3ec9e1e2a28e1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d952f5b95ced5f544dbba258ee4006e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73419629fa67b2e20e8e9f48b90ceda2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae0576dc28522e90776dcfd03bcf2a51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce71befa51ab01917cb1741604182423(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5de964f5ffe82962d92ed0fd28c8b13b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b141862cdff3f6f19d1ed68fec008531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8541349ca1370df01ba4640b4330e711(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb31ad365aa4ae58ea94140dade65072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68d5f3e52a0573b06cbd54e216748a37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2f8983a46d17003226bda2197d2fb56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9c592c543e000cf1731659cf9933688(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ea6825e67c5be8c9da051532d99450c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_590a0d70e3940e03c96887d466df5b31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f409a37541e1a042ba69d59bef56496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_179772b42e19707a24169c0732768071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2499207cee0abf5a3e0e234b3b826d3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e124e24e662f27dbdf42153b0546976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_beb8c934b384a32de12a6e2c95ddabd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9802bbdfdbc5c09d91029868736f758(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5472f6622444f324db0769ba750d9be1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddaaf6d007f863b9c14c6280c1dc1a7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3134890cf9a53cda34a322aaa5972ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e05354b903d65796fdfc3509f681ae7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60b02d35eb071322d6238ad534758a22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94d48adf9c9534ca9fd99f2a6881543d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0bd75fe47ca8ac068fe6f443a11a050(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f77cc741195d42ffacea80cbd6955c2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_446c14f7e3e1a3dc8518d555dac202e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbb09169e39ea07a7b43a294317b1e4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5411fbadd47e47a35bb5884aa6bd3010(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea020fa864a1c3d5973e6ef4c8a6986b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_250e252ac3e99140459d6e567f63cf4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_819b0e96b20559cf5ded0c717720d7ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f95bbe89cdf1f6dcd8777a3aae4d0bf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b9229eba604aea84afd4724f4d4d406(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb0a5e19cf0fb6a9119261f3d3f3b356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53df56ce5fb7768b17794d83ee90e96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d137ec60cf82b7fc7a7eb539c05790c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a94c4b026317e70ebb143db8135f6973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb0a5e19cf0fb6a9119261f3d3f3b356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b8d5d446083e75442f7a34849811d5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e59dc30e576a66614d811e98d2667fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd808bd40a08e8e92a5729663d4719b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9f31692fae795e0b7ac7e568473bcf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b8d5d446083e75442f7a34849811d5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb31ad365aa4ae58ea94140dade65072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68d5f3e52a0573b06cbd54e216748a37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a6d1818f673c290d20f9b1b8f5fa478(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_677996df9338359e68ede0e1f1df7b48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86401ee7f694967208f59e15baa179ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddaaf6d007f863b9c14c6280c1dc1a7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4648fca813e1dfe366f63f16e634914e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f415271ad99bb696c326169d58e4664(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e2c063f4424038e6efd00575240c2e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c317c393fc4a26fd92ffe63d298461f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d70e964be92ae1a4b85b574d5b4c2696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4d0ff14d34d4f26b1aa209eb95f37e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93d2995b7195264788fea5ab5e9ac6ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71b921c5b53a3b4783e64ae8a8378cef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b1a842133562d6336d8076eb38887d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d62bfa8602c29182a2eb60864a1f373a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93d2995b7195264788fea5ab5e9ac6ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c17fbc9e27bbcfa4b32cc1a779fc407(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67511dd00aabb1a4d5ee8810d62acd60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_633cf9a8bfd6bbe35782a786a89b73f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bd8dc09981592fa276d775661e90aa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd6b03ab2cc329b70633abef53c1f136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 40, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 6625], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0e2e350ee5e2d0503c8a3ad12c5517c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f68525246a2dc8f705f887c3fc02f5ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1708a94c5fae8d1dde586aee22b1e8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1deb090241a0d1f23be31c92c50c1214(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_652be3632dc9492f665fc9ed672f861d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b507d01d1ba263d135bc7506cb12e05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d440170c1611e9c7fc6ede878e476b53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbdb5566372b512570e239f8ac0fedbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_652be3632dc9492f665fc9ed672f861d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_446c14f7e3e1a3dc8518d555dac202e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbb09169e39ea07a7b43a294317b1e4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f39e9d9d1a0943e21cd10b18ea93cf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b5385c7b3abd4a72f93ba5edf02df1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66434977041c9cfd553d6593959853d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d97bc2feb7e6bd3a6921d68b8561db0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26f14d291947578827588495b611031f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f86a4da0eb1fdd6ca6bdaf635e446896(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18d65dc485d241cad08d461ae3d605f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_786fecd7480cbe4cd754eded534e632b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9afabe1e9217c42123358056b302e062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d1c86810e0a2b8a10f06c07cc0c3e46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_694b550ddb96d831e480ee0578d7a7de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83fca9c56d57b15bb22db32e671b9942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04afd0f18bdfa46d547463ba4f36182c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d1c86810e0a2b8a10f06c07cc0c3e46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5dfef2c828cbefc71203931913982b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_741a95011f5dd9d427c0f112a617d1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90515c4e314d15f7851b26b36a208381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66752db4dbb53cce759829d3537ce831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5da8420559b7f3490a7073c6e309937f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_741a95011f5dd9d427c0f112a617d1d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e03ba3d0fd319d45fa67df70affa1bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.uniform([1248, 312], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6708d86def61d0f95f0827461ef682d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
                paddle.uniform([312, 1248], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ec94bc6403e49369c08e1b59adbf27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1d9af5572e4dc8356ed8a479dd7fba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74e8192c5c940c975b16722549ea63ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de0cc685fd8a18ef6aa63e0746db6370(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c54e9927f86d41858a5e3b7d54a970ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_432db4eff8fb7b7739f17d96c4dc53c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10814af89ef6a038961af01754a7a59a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_432db4eff8fb7b7739f17d96c4dc53c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83c9cafc0f8477b8b25a1c095d788c7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69ea57c83bbaadcbb4cc81b23db436a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f87feb7e8a3219b639dff4f846fbc3f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb5260298f5460bb2216efb726666789(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10814af89ef6a038961af01754a7a59a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99f5a70fca6f7b148d24219dc56a62c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 39], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dfd91832853c3a3e90abbdac0bc891a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
                paddle.uniform([39, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f01c6510ceae05be4e41df40f5b17925(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d30ad99157f5feec99219f6947ce880f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99c561b9e6a2740c7bd644cb24563808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a6e29713e428ec32f3ff73ddc3d5ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f01c6510ceae05be4e41df40f5b17925(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33e15a8a206d2b02c9a72552c4eee186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dc1ec4da69ab9134e96843c3d4a3f35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b518c7cdf6bb831db3801d6afdc71dfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9afabe1e9217c42123358056b302e062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5df0cd8a35fe916551124c98d2cebae
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f2dcdfb2d4cd45d9043a14b0abf54d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce87bbe98688938f73d8668cab9c07f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c07eff1d31ccc5326f8ca5f887a3bafb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ebb73301f9b4de9d30e2ac7ac1582cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_152bcfffd2818c25659dff7aa9ad3ab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_548b84edf6d59eab422ac8f099676566(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83f06b0b73a58fb555cabde5e1289b1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d9ca86ffeda060988e6224fc350231f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 25, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 37], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8d006e4f898329c113744fbae2698f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa49dbc84c9c52da3a21ba51a93af549(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41ca8148577acf1813f75efeb881ef91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_874b2ca960113b9c5e02fcdc6b44b67c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[8.870004653930664, 9.070971488952637, 8.6998291015625, 8.771418571472168, 8.77302074432373, 8.278258323669434, 9.320741653442383, 8.51421070098877, 8.791091918945312, 8.457852363586426, 9.51916790008545, 8.142207145690918, 8.19686508178711, 8.17872428894043, 8.666762351989746, 8.191591262817383, 9.497175216674805, 8.180706024169922, 8.71663761138916, 9.759553909301758, 8.672922134399414, 8.273152351379395, 8.350447654724121, 9.068215370178223, 8.876349449157715, 8.625724792480469, 9.934979438781738, 8.560579299926758, 8.065917015075684, 9.131192207336426]], dtype='float32').reshape([1, 30]),
                paddle.uniform([30, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff20a947232a3d8a7dc68bcf9a921ed3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa265265554b13f1410c68ced3e88e10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45e4e5c125a827046b60f3f7e8d207ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_52ef6cbc92bf01bb21a363459b1238b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff20a947232a3d8a7dc68bcf9a921ed3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8abd5965be7fd4ac93e7e2e757a3cee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_588803e98642cda81a1f8fbd1438d888(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaae33f1a06d14aeb34131e728fc9b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31e0c683c23bdee189900458a077cda9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8abd5965be7fd4ac93e7e2e757a3cee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f95bbe89cdf1f6dcd8777a3aae4d0bf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b9229eba604aea84afd4724f4d4d406(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0772e7ad90e364b830cc09c0045e5001(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9398fecb73cf9b706d832802793019c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_204a793f8c6c0a581231dd278be32213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa360babf74baa0cea6c562f321c5a35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe1ea407bf9108a38750edcde4000a85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58d753211af4743d964f15072e7381ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_204a793f8c6c0a581231dd278be32213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b9f71882be5f94c173e1cdd736f97ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f798ebefb387e047a3b072274e05f7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e51f55740fd190c47d8a51b2ca8d597(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f2728c9522bd461b430038f5456f104(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_010d25ae262d6943f3dc15dc1c0a6b51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12af8f64e4d8d3b67d3ec9e1e2a28e1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c91c109e7bd99559f0a1d4708bc31488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c2c2a00506c6acc95d61dd9cf553fd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad93a227e1fc91491a66358166e16285(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3f00bc0aea766392c46e33f874bcf00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6044757c80d11389dd49fd83b02d0cf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c2c2a00506c6acc95d61dd9cf553fd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b0223290dbc13b1b34b4c6fa125e1b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b507d01d1ba263d135bc7506cb12e05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd51256176c885521a8e83cf36fa8ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_076d99e8f46a8e4e9f0a9982a38f89ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b0223290dbc13b1b34b4c6fa125e1b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7adcf8cfbe0c24004d726bd2bf9ef9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a57d61c6d5235e2e5fc478115c1da014(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e190037110145f45841c93543eb9114(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f549242227fc8177170c5df07938d111(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdeafecb329fed164e65e6dd8965e80e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2e7b329b9efa44d01281371ced86827(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_086a27325c7a62ab93361e21cf94d642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ada90fb5c3ec1d6e106a19a7ac4dc9e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdeafecb329fed164e65e6dd8965e80e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a03875ae41e0e8300ccd6474b2feb55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2dd14d8e2cf55482d0409cb020789b63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d221b18dcc4dab04e9c948845da11e47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_079ac77e23278e045cd3f61f3d4b8ab5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c8ff2d5f3264a5986b36d74b45d8ccb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e015beb941795e44e392715e6680755f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83357e70e5a5c34331f47b3120441b2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdfbab05157498ca21e5790e0f9d373c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_679a0ea4616e8af3b34ec74c19578980(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91116a2dd3318d24e5b2d8d8368445f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8cf912dda3b4653538686b0d13a97bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00e3152baf039045555807497af95eb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_963993f752682a28d948d8942c9a1592(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0644b416971eb804c0c5dface3d0daf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1608bd5ef80319dda7081d2d6bf8d14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad93a227e1fc91491a66358166e16285(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f87a5253569f8341ef1bd141d7a9c6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e597b626b77cd1991d87c6190d36edd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1608bd5ef80319dda7081d2d6bf8d14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a0e40bc2fd6d8aeefca06cd5b1ffefa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.uniform([624, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f21e6024f6e3bbc24611515c427caed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed8a7671950074cc73ed04519a0c1cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c23ae625db27972903d7a6f27b841df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2928b61c4899a1630afed121defab0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_572bc33a889390428e961a8a115c3e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d09b46ede49a01864a0d3fcec4f9a883
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35acd718f8b1e20ea4e4c6881e6d7dda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f385054aac7d6b38c59dfc857791c857
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()