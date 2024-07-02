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
        paddle.seed(2024)
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
    class PrimitiveOp_834c89880d6e77b4d9179f618594d3ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9182af76a46cddd7ab699d0d10a5e444(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_834c89880d6e77b4d9179f618594d3ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
                paddle.uniform([91], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e8dac5f456994979340dbf976a4fe3fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d3bb262bc3f67fe557625fba1f08676(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8dac5f456994979340dbf976a4fe3fc
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57000913d2634a9eb94f6c6a99a8332f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_471f81a7bd7d34553934d288ac9543ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57000913d2634a9eb94f6c6a99a8332f
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_080a58cebd644135cc1e10c9f224231a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 8, 8, 7, 7, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcfd0eca09ee719ca0744169dc20e70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_080a58cebd644135cc1e10c9f224231a
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8aac5687f411eef5b2363a14bc6efd02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[72, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_66630c9ecd094b9cc9d6fc290206da85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aac5687f411eef5b2363a14bc6efd02
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_18db447991dedc55780212b15c9bd1ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 18], dtype='float32'),
                paddle.static.InputSpec(shape=[18, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62799611c88aa3b889f72750782f3ddc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18db447991dedc55780212b15c9bd1ea
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.512653350830078, 5.044649124145508, 4.655038356781006, 4.622213363647461, 4.632609844207764, 3.9768052101135254, 5.023960113525391, 4.7768354415893555, 5.421357154846191, 4.686634063720703, 5.148271083831787, 3.8738279342651367, 5.151643753051758, 4.582184314727783, 4.700376987457275, 4.343968868255615, 4.77421760559082, 4.917938232421875]], dtype='float32').reshape([1, 18]),
                paddle.uniform([18, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95ced37c9ea5ef78048dee6d5ca23778(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5334e7e477a5ce72b10c59e14b86ced6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95ced37c9ea5ef78048dee6d5ca23778
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_332879d10382dfa8850f36c27f26707f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca94f1a2c29129f2daeb913cf4678263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_332879d10382dfa8850f36c27f26707f
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d2e66cb9f65307710ccd1a02dadd61a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a56b241effdb8304222e6621a4a965f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d2e66cb9f65307710ccd1a02dadd61a
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_030a8c3cf557782c37e21593ced83cb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0053bc8277d8b154f67581aca10b2b8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_030a8c3cf557782c37e21593ced83cb2
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b8dde8e9b6d331dbbee7f000ba7a4229(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[92, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ff13f9e5281c1533f1722728e1a4afe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8dde8e9b6d331dbbee7f000ba7a4229
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.uniform([92, 23], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8907bca9eb1a3ff9b3e19e9925d7441d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 23], dtype='float32'),
                paddle.static.InputSpec(shape=[23, 92], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_914e6c7d83ed53b161b5c5602b809d75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8907bca9eb1a3ff9b3e19e9925d7441d
        def get_inputs(self):
            return [
                paddle.to_tensor([[6.194968223571777, 5.21784782409668, 6.012714385986328, 4.90752649307251, 5.907154083251953, 5.492164611816406, 5.724031448364258, 5.579870223999023, 5.25261116027832, 5.262575149536133, 4.954087257385254, 5.3679680824279785, 4.764451503753662, 5.245046615600586, 6.278081893920898, 5.532181262969971, 6.717957496643066, 5.762195110321045, 5.715753078460693, 5.413527965545654, 4.952664375305176, 5.497104167938232, 5.146821022033691]], dtype='float32').reshape([1, 23]),
                paddle.uniform([23, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f68c5d26ee0a68b510cc5fca64678ccf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae1e17aaae2d830fecc35a797b4b3023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f68c5d26ee0a68b510cc5fca64678ccf
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c41eaced530f56d3d75d63871561bb79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 3, 64, 198], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6188afb72fc08a385262f9fa62afd54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c41eaced530f56d3d75d63871561bb79
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fffe5708b45848ccfc52c1442b8026e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 198, 198], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 3, 198, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87ec07735e6de39024c4d86168a9e07f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fffe5708b45848ccfc52c1442b8026e7
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_70f0967dc29eeb31030dc16a6314a387(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 198, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_856dfeccfc7bdc500438e0d5dcc7a07f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70f0967dc29eeb31030dc16a6314a387
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_67a836352b910bcea26d4b4a1eabe300(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7a154bb9b4bceed2bbc5d596f81a253(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67a836352b910bcea26d4b4a1eabe300
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_957c9dbe734d86f482d9b4b911a1c807(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8514f35ecfa5168942d2bb73b4a6262(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_957c9dbe734d86f482d9b4b911a1c807
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c23b884349f4958ffa0efe8706a51d2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2048, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21953445e03b793caed1f1f970abd133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23b884349f4958ffa0efe8706a51d2f
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5df3c20cb42b9a7a2bf1371f92d5cee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8dac5f456994979340dbf976a4fe3fc
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_abbc7ad398f7168b69b6faaed2ffc26f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4ae57cca9c696f0a380b56c3fb8737d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abbc7ad398f7168b69b6faaed2ffc26f
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_50df5e7dc75e0bafc8685d5ed205daad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[960, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc396d8fd2d1b4eafbc8cbd7798df641(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50df5e7dc75e0bafc8685d5ed205daad
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([960, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2bd9b6aa1f43b3598977b59bce324a93(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 960], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5082ee349cb646bfaf2722790fae6133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bd9b6aa1f43b3598977b59bce324a93
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 960], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0163ca9e65e3b0c7dc77b6d5fbe9c193(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[480, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8777efb32dfd79e08998d8fe3b75a3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0163ca9e65e3b0c7dc77b6d5fbe9c193
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3e35db49230b16435cdfdcf9dd0d8ede(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 480], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0138b718244b4add262f4bff22b8f5f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e35db49230b16435cdfdcf9dd0d8ede
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef71b4937ca7a57a5c678404e8689491(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[12544, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01cb7f8cd46b070d70b87f543450e093(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef71b4937ca7a57a5c678404e8689491
        def get_inputs(self):
            return [
                paddle.uniform([512, 12544], dtype='float32', min=0, max=0.5),
                paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aad6c49551d701a0ff5d66e617391ddd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7083fd8a7dbbf78ea94b26184cc8f05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aad6c49551d701a0ff5d66e617391ddd
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74adb34a00ab789439d8645a2c4dd255(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[336, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07c10885fb75a6c876b97c6a8244e77e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74adb34a00ab789439d8645a2c4dd255
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23d3aa925ac24fba3989381380ca7788(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 84], dtype='float32'),
                paddle.static.InputSpec(shape=[84, 336], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8a6ee0b4942e9594e732c97e5160d19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d3aa925ac24fba3989381380ca7788
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2e4a7d21823bfee6878e36c926c29ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 8, 8, 7, 7, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ba6a00c2bab6186a186d6a61e762caa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2e4a7d21823bfee6878e36c926c29ad
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1e16a99b0095e07e0185243d09bbdbf3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5e20259201d7296a56047cbe367db95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e16a99b0095e07e0185243d09bbdbf3
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6cc5e02a3f5bcc30d7a1622513286cfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e16a99b0095e07e0185243d09bbdbf3
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa293663fcc6a6d2a756b47de9443f0a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3989d51e24c2d9a3742577b18fe0d0e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa293663fcc6a6d2a756b47de9443f0a
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_359e23b542fa225927c92dc7c3329024(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a5783058a3d13f83ced7732a90bfcce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_359e23b542fa225927c92dc7c3329024
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4e48c67547d4a15f47bb9014c7f46dbc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[15, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e372234e3f577c3759dc5403eebef726(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e48c67547d4a15f47bb9014c7f46dbc
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_509e1178fe258a00ebed31a79146f8f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74adb34a00ab789439d8645a2c4dd255
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8ad814c584dd930fd4e86d3029934dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d3aa925ac24fba3989381380ca7788
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44d1342dd18779d503d7f37a55541af8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23b884349f4958ffa0efe8706a51d2f
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_509e1178fe258a00ebed31a79146f8f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74adb34a00ab789439d8645a2c4dd255
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8ad814c584dd930fd4e86d3029934dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d3aa925ac24fba3989381380ca7788
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ab822c6716aaffb980b6765c7bf19080(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd1f1684fe2fd7f75bceaf4fd5a19475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab822c6716aaffb980b6765c7bf19080
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_38b3c9fc51d416f4a21066b220601a8c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c7ee555f33f849d98c7a5495d255713d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38b3c9fc51d416f4a21066b220601a8c
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_916408ccdb82f84fb5def8e558c17d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95ced37c9ea5ef78048dee6d5ca23778
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ed484e01321e6f5a190e7f6e7d0cad0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_332879d10382dfa8850f36c27f26707f
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78278614dd91148c277c16214a4a7ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d2e66cb9f65307710ccd1a02dadd61a
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a16fd33eafe176ebda82bba775ccc08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_030a8c3cf557782c37e21593ced83cb2
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_30db99a4f1abc3e205fe20057f8743ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9faf8a08fbe22070f4b430ada4dfc4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30db99a4f1abc3e205fe20057f8743ec
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f25209c060f711efbc1db89db49cb0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[60, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20a659c824aa4ca7ca8c61a4315e2dea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f25209c060f711efbc1db89db49cb0e
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_51b4104d311a25dc4000de215f8c973a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 4, 4, 7, 7, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8f450d580e09ccaed90fbe677fa01a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b4104d311a25dc4000de215f8c973a
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f9b6717e515acead502c5f2dd7f62214(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be75d7f670490af38c2617879e8e3c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9b6717e515acead502c5f2dd7f62214
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc71f981778a49f29ada0529668d2d58(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29c9becc49a980f693ba65ec00b60891(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc71f981778a49f29ada0529668d2d58
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 577], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_30a9c9205612f91fedb5e035791dae72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_864074cb641e07d6d0f11780bf90a79f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30a9c9205612f91fedb5e035791dae72
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fc5cd4f845f036f61093a2165184ccbd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1c8c6043a6a741a9fe7de2b1deb91d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc5cd4f845f036f61093a2165184ccbd
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce3af1315c6d51f3589ccce508d03835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_359e23b542fa225927c92dc7c3329024
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0440e5360e508fa84935688a8da38e83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e48c67547d4a15f47bb9014c7f46dbc
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ca1c9efa2a7926a05ad71a0726627c72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c6a9ec4d8a5e5b4a96f0f7984329722(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca1c9efa2a7926a05ad71a0726627c72
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20598eef8c10158dfbd5eed88bff64d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e16a99b0095e07e0185243d09bbdbf3
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0c3d47c17fade3bacc286be9f2b65086(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[872, 218], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab87c7a3ed638bd876844dbc9a303930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3d47c17fade3bacc286be9f2b65086
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac97e46c97fbf8e8c5b9c782718df1bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 218], dtype='float32'),
                paddle.static.InputSpec(shape=[218, 872], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a2d5cf1b56d859e762466607827438c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac97e46c97fbf8e8c5b9c782718df1bd
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4bba3688b32fb6d90871725772fa3c64(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 4, 4, 7, 7, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1c71fd0e5965aaef70b7745fbafc593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bba3688b32fb6d90871725772fa3c64
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d668e91ed89a78c41da1a9fd1e4dac4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a21731d5c5371910cc6508ea5665b16b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d668e91ed89a78c41da1a9fd1e4dac4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7b7e64a6f506ecd18102e2d30bdd2c38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e13f23f4e64705c5e6aa39c2b042ffbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b7e64a6f506ecd18102e2d30bdd2c38
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e05f5f6ef70cd3feab9b1016d7730eec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cce2aa1296636a5ad7c6d8f5dbceb629(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e05f5f6ef70cd3feab9b1016d7730eec
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fcd38c295e387beebab4d21107b65a62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e903ca4017ad173a2931e14bf5779ad2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd38c295e387beebab4d21107b65a62
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a21731d5c5371910cc6508ea5665b16b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d668e91ed89a78c41da1a9fd1e4dac4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd1f1684fe2fd7f75bceaf4fd5a19475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab822c6716aaffb980b6765c7bf19080
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7ee555f33f849d98c7a5495d255713d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38b3c9fc51d416f4a21066b220601a8c
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0bda6c2b29a0d77efbca213c91271a8d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9793a904dfc8937f0f4e95fd3b2ec007(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bda6c2b29a0d77efbca213c91271a8d
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_602cb6f83f46846b9ae579ca28c29757(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 1536], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_79e1d4726567ae06215f3d71c4a44f22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_602cb6f83f46846b9ae579ca28c29757
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5f3f4201322ebce7f6c0c61f49efa21c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[3136, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5375cc4d07e215a411bec87729edaef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f3f4201322ebce7f6c0c61f49efa21c
        def get_inputs(self):
            return [
                paddle.uniform([390, 3136], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3bbbb3274892e050d5dec9d0e4b819d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aad6c49551d701a0ff5d66e617391ddd
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_039edc5d08bcbc812f862f26a6fc03d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74adb34a00ab789439d8645a2c4dd255
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2af6be46fb2cd657b8799cadc35569b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d3aa925ac24fba3989381380ca7788
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5d649301c77cad3d27ba54257f91009(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5112bec7f3444122ef82c0140a9b676b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5d649301c77cad3d27ba54257f91009
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3e2706c0a40080a3c65470a6308aa5de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f961efaa63009b7335234d357ed89f4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e2706c0a40080a3c65470a6308aa5de
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1cd606756e45f10541d35d5922807517(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_707162d6adf4347dd655b90116c13369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cd606756e45f10541d35d5922807517
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ff8a9a182a08eff47f49b40fbc62bf94(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dfedd7c41f9b5f58f09b3216dfd0e2ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff8a9a182a08eff47f49b40fbc62bf94
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_369c500c294d5fd9a38c2e3e70644f28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30db99a4f1abc3e205fe20057f8743ec
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5f1e064ef34ea90572b9faac808dbc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f25209c060f711efbc1db89db49cb0e
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c6e06e783fb945a6ac3d61942f6f57b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f68c5d26ee0a68b510cc5fca64678ccf
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44a7af4a50fd9191d808715d91595a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c41eaced530f56d3d75d63871561bb79
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6f5db0662da0ac3d4c89d13438cba64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fffe5708b45848ccfc52c1442b8026e7
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dfb567306c6bc063fc66255a0202c10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70f0967dc29eeb31030dc16a6314a387
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85416f3dd36eec865e16978528e7b754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab822c6716aaffb980b6765c7bf19080
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6151efe3087bbd37ab5178df86d9dcc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e9b6932b6d5eaef0a8ef49f77884768b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6151efe3087bbd37ab5178df86d9dcc
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_574ccee023408081a3bd3c7c1f6fd367(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_655a31a382ccae385cc0e786784d7f5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_574ccee023408081a3bd3c7c1f6fd367
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_655a31a382ccae385cc0e786784d7f5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_574ccee023408081a3bd3c7c1f6fd367
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b263b870cdeaa4e1327a5efa36a82786(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1dc491182902794575b67a12df53c3a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b263b870cdeaa4e1327a5efa36a82786
        def get_inputs(self):
            return [
                paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b825d16400b94ba186f839f357489382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_574ccee023408081a3bd3c7c1f6fd367
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b825d16400b94ba186f839f357489382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_574ccee023408081a3bd3c7c1f6fd367
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4cd9e2a7e0f82023be3f7a351611032(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 2048], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_050bf227a2ee88b529580bfeac2e61e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4cd9e2a7e0f82023be3f7a351611032
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 2048], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a164dbd5d81914ffdda01ae275d43a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2048], dtype='float32'),
                paddle.static.InputSpec(shape=[2048, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88056d18087ef307c292f76e87cd8a40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a164dbd5d81914ffdda01ae275d43a7
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_25303d287ea951bc4751b32870e3e72e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1, 1, 7, 7, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e4da86cbc6a8d357c521cd67cdff1f27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_25303d287ea951bc4751b32870e3e72e
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_515c619f002723757f902f6a09f3b61b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67a836352b910bcea26d4b4a1eabe300
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2cb11cc407de9ea7bf43e2cc770c6a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_957c9dbe734d86f482d9b4b911a1c807
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07c10885fb75a6c876b97c6a8244e77e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74adb34a00ab789439d8645a2c4dd255
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8a6ee0b4942e9594e732c97e5160d19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d3aa925ac24fba3989381380ca7788
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ba6a00c2bab6186a186d6a61e762caa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2e4a7d21823bfee6878e36c926c29ad
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_125498ec42cceefebd9072935b20bb43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23b884349f4958ffa0efe8706a51d2f
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_67b8d4299af2db5931e65e6b59438f1c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[36, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e6c82236d1750cc8e09370969a57365(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67b8d4299af2db5931e65e6b59438f1c
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5cf1eb2d5badd705c4bbd8d69975dea2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 9], dtype='float32'),
                paddle.static.InputSpec(shape=[9, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11911b96d7552c8a558ceda78be7ad6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cf1eb2d5badd705c4bbd8d69975dea2
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_69bb3ad8a418a98695c6a4535d4ad111(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83e88c7c03059506dae344a0a1edf222(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69bb3ad8a418a98695c6a4535d4ad111
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_58fdb16bc745ff1b41c92a915510aacf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80824d284ba14ad5c5b7d9932e8caa64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58fdb16bc745ff1b41c92a915510aacf
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_03b6ef95c82ac7e22d53e37032e16780(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07913922860f237cfd3e63fccacdb836(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03b6ef95c82ac7e22d53e37032e16780
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_097a882373cd37cec13c19223ac87bd8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a054568bdea527cca9e67362223b15d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_097a882373cd37cec13c19223ac87bd8
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83e88c7c03059506dae344a0a1edf222(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69bb3ad8a418a98695c6a4535d4ad111
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e89383b3679da523023ff28957d1d8c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1280, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ce7135133777ddfaee7663a727518d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e89383b3679da523023ff28957d1d8c9
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67b903bd1f2c513dbcdfbd64871fc6fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e16a99b0095e07e0185243d09bbdbf3
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_56e154929774af06bda3433d5aeea717(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f560ee694e9a7612abe9fcacf64aa19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56e154929774af06bda3433d5aeea717
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5334e7e477a5ce72b10c59e14b86ced6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95ced37c9ea5ef78048dee6d5ca23778
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca94f1a2c29129f2daeb913cf4678263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_332879d10382dfa8850f36c27f26707f
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a56b241effdb8304222e6621a4a965f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d2e66cb9f65307710ccd1a02dadd61a
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0053bc8277d8b154f67581aca10b2b8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_030a8c3cf557782c37e21593ced83cb2
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc38c5a6d8340819c661fc1387923b1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0163ca9e65e3b0c7dc77b6d5fbe9c193
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a85215fb19d00c48efbac238797eecdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e35db49230b16435cdfdcf9dd0d8ede
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbb2591fd58824322d9b2a95f15a297a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9b6717e515acead502c5f2dd7f62214
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc5a8610ad711c64c6debb762f0090f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74adb34a00ab789439d8645a2c4dd255
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ac2635659bde6efbc9b7f3be8d33096(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d3aa925ac24fba3989381380ca7788
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_524fd3c6ff48d4d1bf939321a48f7cf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30db99a4f1abc3e205fe20057f8743ec
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4460f66c4607ade6d11a4c055b5c7534(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f25209c060f711efbc1db89db49cb0e
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_039edc5d08bcbc812f862f26a6fc03d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74adb34a00ab789439d8645a2c4dd255
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2af6be46fb2cd657b8799cadc35569b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d3aa925ac24fba3989381380ca7788
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc6eeeca53cba6579c1fefadf656ae53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1536, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df6d815e01874a09b0aa1795b86a282e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc6eeeca53cba6579c1fefadf656ae53
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d060db95b30962a32592d8537de69a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_359e23b542fa225927c92dc7c3329024
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27ebd83f25f669c543ff4f7b0fd23963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e48c67547d4a15f47bb9014c7f46dbc
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfce6205555c13cefd2f97ddf493a8ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30db99a4f1abc3e205fe20057f8743ec
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e112d9eddeb342fc595c9541f0bc7e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f25209c060f711efbc1db89db49cb0e
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b44a9c0adabe50fca27e8699b1f35d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc6eeeca53cba6579c1fefadf656ae53
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1c71fd0e5965aaef70b7745fbafc593(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bba3688b32fb6d90871725772fa3c64
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21953445e03b793caed1f1f970abd133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23b884349f4958ffa0efe8706a51d2f
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c11217327c69c7783553097801acc09b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55ee7024fc895ef2a9b5af5059352c7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c11217327c69c7783553097801acc09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ef395254f3623bf4353c14b1f23efbf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 6, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e3ba9c46a536b5904c244a6005b268f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ef395254f3623bf4353c14b1f23efbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95d86f06c7141c58553c42c908a9db79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 6, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b39e35808ec446949c03be87cf9575b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95d86f06c7141c58553c42c908a9db79
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_884b2c3cd6c15e07cae6e3ad52287c16(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e67cbb916f0e4a1edf2cf0444c516a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_884b2c3cd6c15e07cae6e3ad52287c16
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b8585f549217fc6f615269ebeba33f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67b8d4299af2db5931e65e6b59438f1c
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7ed6f5c2dc73a7ee729eb4291fd0b9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cf1eb2d5badd705c4bbd8d69975dea2
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19d2651364ece9ec405dc7f44a4dace5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bda6c2b29a0d77efbca213c91271a8d
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daa0a5d1938d548556664460d9675bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_602cb6f83f46846b9ae579ca28c29757
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9b2b2a6cb88083b3be584f8e421c6224(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 150], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ce0f8e994a01ee7ddc788d7ebcda862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b2b2a6cb88083b3be584f8e421c6224
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ba118f07911ea2d51182e059ff25d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c11217327c69c7783553097801acc09b
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac11853f25f8bc0df5ced6c504386afb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_359e23b542fa225927c92dc7c3329024
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cb8a7fbbf77891b73523b4712e56835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e48c67547d4a15f47bb9014c7f46dbc
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_60d5535b91d4909aa070b4cfcaae835f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[672, 168], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c59574e021f8c4dd6349e9baeff3f00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60d5535b91d4909aa070b4cfcaae835f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3ae86b4d3b7d237d1533ba8a8d51e11(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 168], dtype='float32'),
                paddle.static.InputSpec(shape=[168, 672], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e6dcb5a90c4fe8adc2c44ca14a6d9818(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3ae86b4d3b7d237d1533ba8a8d51e11
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e29173f2b56217eda3e8f3542085e537(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c659c9acdf80379b238fec580aaa6c6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e29173f2b56217eda3e8f3542085e537
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef6fad89881b113f26334b36d3b9dabb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8eff1b4332476e1875909b7ba7b2fa07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef6fad89881b113f26334b36d3b9dabb
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_160b36e35122efef03af10314171eb1a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2618eaaf23fc7c83d74de99d50f26440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_160b36e35122efef03af10314171eb1a
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2b19c29071d42918f7f7d4eba389cced(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f01d333557defcb9cc8b53e1d6b17b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b19c29071d42918f7f7d4eba389cced
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c659c9acdf80379b238fec580aaa6c6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e29173f2b56217eda3e8f3542085e537
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0593035859a1b2656ac852798b1278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69bb3ad8a418a98695c6a4535d4ad111
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db85a1954f993460bc2ffe9f58ba9bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58fdb16bc745ff1b41c92a915510aacf
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e2f96f4737af38e61b17d2e8b5f965c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03b6ef95c82ac7e22d53e37032e16780
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f38cb574d05613df171003f16c2a9fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_097a882373cd37cec13c19223ac87bd8
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0593035859a1b2656ac852798b1278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69bb3ad8a418a98695c6a4535d4ad111
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc5a8610ad711c64c6debb762f0090f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74adb34a00ab789439d8645a2c4dd255
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ac2635659bde6efbc9b7f3be8d33096(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23d3aa925ac24fba3989381380ca7788
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d3bb262bc3f67fe557625fba1f08676(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8dac5f456994979340dbf976a4fe3fc
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_471f81a7bd7d34553934d288ac9543ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57000913d2634a9eb94f6c6a99a8332f
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e2a36ef45bdf1c7c4d033729e89e6e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23b884349f4958ffa0efe8706a51d2f
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21953445e03b793caed1f1f970abd133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c23b884349f4958ffa0efe8706a51d2f
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7d183a2c01a73f05691d185fab70a9cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74545e7d5b9ec5cea0e74df28f8fef89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d183a2c01a73f05691d185fab70a9cc
        def get_inputs(self):
            return [
                paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4da86cbc6a8d357c521cd67cdff1f27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_25303d287ea951bc4751b32870e3e72e
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14cc1060cba795777182d20a04d1c707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f68c5d26ee0a68b510cc5fca64678ccf
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da3ab7333c101aa3a0437bb99d8cec6e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 3, 64, 197], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bca1d71a299e234cac9c23312cc63f3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da3ab7333c101aa3a0437bb99d8cec6e
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9f22e51c8402c54e57dc6f22891f055(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 197, 197], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 3, 197, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f298951bcfebdb6c00784e11a1ccdddb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f22e51c8402c54e57dc6f22891f055
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_96e2f107b83b1032428957b0ad605134(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 197, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e86b501566ad85c87826709e85e978a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96e2f107b83b1032428957b0ad605134
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a718811a7380b4d49d9f5fec9723dcfb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87be5e577e511e4172472be886a88a8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a718811a7380b4d49d9f5fec9723dcfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a4367ee347a6599aeedf21da828a7ab7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c16cc4372d078c2380c1171e22e835af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4367ee347a6599aeedf21da828a7ab7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c82f110c54ea53f84f35af01db95d8e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e4b4215cae643bfade00f40cf6f17a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c82f110c54ea53f84f35af01db95d8e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b1e2e738fd8a181f8937a1ab2107f4da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_457af43fb0b8c74f3dcb479a4e894375(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1e2e738fd8a181f8937a1ab2107f4da
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87be5e577e511e4172472be886a88a8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a718811a7380b4d49d9f5fec9723dcfb
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8735b7e2edeafa73cd1cbb090ce7ddee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f68c5d26ee0a68b510cc5fca64678ccf
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8f450d580e09ccaed90fbe677fa01a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51b4104d311a25dc4000de215f8c973a
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85416f3dd36eec865e16978528e7b754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab822c6716aaffb980b6765c7bf19080
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9b6932b6d5eaef0a8ef49f77884768b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6151efe3087bbd37ab5178df86d9dcc
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c5b65c806da86f612ca9349df2d6203f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 6625], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be079a93c4eb2ab805e179fe11018d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5b65c806da86f612ca9349df2d6203f
        def get_inputs(self):
            return [
                paddle.uniform([10, 40, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 6625], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_916408ccdb82f84fb5def8e558c17d39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95ced37c9ea5ef78048dee6d5ca23778
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ed484e01321e6f5a190e7f6e7d0cad0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_332879d10382dfa8850f36c27f26707f
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78278614dd91148c277c16214a4a7ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d2e66cb9f65307710ccd1a02dadd61a
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a16fd33eafe176ebda82bba775ccc08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_030a8c3cf557782c37e21593ced83cb2
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f8e4474f0dca8870b6e04d773f907b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b450a13b83b146205c6db23bc4d6dd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8e4474f0dca8870b6e04d773f907b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7f0f3ecb1acf2e6f686c060250d08f64(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a9b8971db27df9375024cce64891048(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0f3ecb1acf2e6f686c060250d08f64
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3784b38f0bf864e2eb7654d8b31bbbd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1046800ff8050af48fb1e8a7c6e8cd10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3784b38f0bf864e2eb7654d8b31bbbd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_de898f85dcb29342314e3ecdc024f032(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41edffb4b1078c5b2e9f315f0d8b3b98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de898f85dcb29342314e3ecdc024f032
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b450a13b83b146205c6db23bc4d6dd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8e4474f0dca8870b6e04d773f907b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19d2651364ece9ec405dc7f44a4dace5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bda6c2b29a0d77efbca213c91271a8d
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daa0a5d1938d548556664460d9675bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_602cb6f83f46846b9ae579ca28c29757
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_606ec4c7d1491b3f0148a7944a39448d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 150], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdf8f9e44cce7edfaa6525dbf71b36c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_606ec4c7d1491b3f0148a7944a39448d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82ca00a5642cd84cb7db55fd94d7f4d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5d649301c77cad3d27ba54257f91009
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e059583732b5dd2ce4db4a0c146ac50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e2706c0a40080a3c65470a6308aa5de
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71eb63f867e15b30273ecd36dacc05ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cd606756e45f10541d35d5922807517
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3a4fce3176a7744b9677b612645958d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff8a9a182a08eff47f49b40fbc62bf94
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8060a7112e6b86995ffee1450218aa0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9b6717e515acead502c5f2dd7f62214
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c63c1b7eba323de67e3e87f090bafb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca1c9efa2a7926a05ad71a0726627c72
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3eef8d27bfa2cb6dba01cb1c87c9984(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e16a99b0095e07e0185243d09bbdbf3
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_43fa6cdc4ac21f3103f7d33c72890199(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 2, 2, 7, 7, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ba806fa61307c588082461886d8a14b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43fa6cdc4ac21f3103f7d33c72890199
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3b82defdaf83fb2064ef9b12a062a6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70fba00f892f9c6fed531ef9d92f039f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3b82defdaf83fb2064ef9b12a062a6f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f8ff1581b71f3856594eeac48d638c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b7e64a6f506ecd18102e2d30bdd2c38
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eb10703d237cf98b1c0f0fca50a6a993(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d25b66a63eb030287f7de1d04eb87389(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb10703d237cf98b1c0f0fca50a6a993
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_995ac51f0f64ce27506e0fb94d190d91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2291113a4a85b410f3e9fc0594f5f24b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_995ac51f0f64ce27506e0fb94d190d91
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70fba00f892f9c6fed531ef9d92f039f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3b82defdaf83fb2064ef9b12a062a6f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0dce33533d9eaf7ae6e97c493b4fe7aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[704, 1000], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84dc34e01485ac2680c063ddefe44a0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dce33533d9eaf7ae6e97c493b4fe7aa
        def get_inputs(self):
            return [
                paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bf2480c8f53084b934cb555dac9d465e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23998b578042a94dae9737d047b6648a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2480c8f53084b934cb555dac9d465e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3fa42a31329352c2a9f5d2cb0bee9bf8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 640], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d17a7637a5a1557e4440d56835ed22b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fa42a31329352c2a9f5d2cb0bee9bf8
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da4ad03476690f1738436014fecd6901(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c5324b07e18bfa1ba7972c43ac26448(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da4ad03476690f1738436014fecd6901
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a4a48d37b73aacc53156accf8dbf4e3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e69ca93a0f7b5f7c663466159d98148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4a48d37b73aacc53156accf8dbf4e3e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23998b578042a94dae9737d047b6648a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf2480c8f53084b934cb555dac9d465e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_894837144aa5049001e9b44254a265b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1248, 312], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ebec6657cef791a2b117937263e17e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_894837144aa5049001e9b44254a265b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.uniform([1248, 312], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_881496b775aefd6f2a907807666ba80b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 312], dtype='float32'),
                paddle.static.InputSpec(shape=[312, 1248], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ce19ee17f5a84b9a5d93b60d62b7a3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_881496b775aefd6f2a907807666ba80b
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
                paddle.uniform([312, 1248], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ec9846672d670e3ef5bbf0b181d44e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bcd353d74853d73645488f5e454ef3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ec9846672d670e3ef5bbf0b181d44e7
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cef064fd8188b0c8a68c93b70d96db8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0163ca9e65e3b0c7dc77b6d5fbe9c193
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c0d35835ad11a3a272c082d65216920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e35db49230b16435cdfdcf9dd0d8ede
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d3a31fd6dc1326c4e0f322e7e2b3144(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67b8d4299af2db5931e65e6b59438f1c
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0259288f1c41b5d280e2f390e0e735d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cf1eb2d5badd705c4bbd8d69975dea2
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f4aa5df6853b284a7e0b8e83eda39074(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1, 1, 7, 7, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a44dbc6774c2bc40af87d3e36b18123c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4aa5df6853b284a7e0b8e83eda39074
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0792d1c032eac973faa1f7a00288a244(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 2, 2, 7, 7, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34233811eefeed52ef898d51c781efde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0792d1c032eac973faa1f7a00288a244
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a44dbc6774c2bc40af87d3e36b18123c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4aa5df6853b284a7e0b8e83eda39074
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc5ac0ede9eb5ac676854264a10ded6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9b6717e515acead502c5f2dd7f62214
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28b253b353a3034da9bf04743959206a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc71f981778a49f29ada0529668d2d58
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a5ca70d70f750359a1f0c89fec6a969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30a9c9205612f91fedb5e035791dae72
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_117d96ccccd666c38068b308229e8d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc5cd4f845f036f61093a2165184ccbd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34233811eefeed52ef898d51c781efde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0792d1c032eac973faa1f7a00288a244
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_050f54110793bbd3c6c0808dfd1b41d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[156, 39], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_746d6c959851315bfdae8451486d5a2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_050f54110793bbd3c6c0808dfd1b41d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 39], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc76cd07acf939827e8e3ad87a906a17(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 39], dtype='float32'),
                paddle.static.InputSpec(shape=[39, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbd544d5ed8e13f6f72b8d804fab687e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc76cd07acf939827e8e3ad87a906a17
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
                paddle.uniform([39, 156], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_974ca4aec87839abdc7818643f5c2125(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bb692de8a63004988cb6897cec3d708(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_974ca4aec87839abdc7818643f5c2125
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c4a69cadd3a653e18a75a5e05bfb80c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5130742ed5f494d6fc2a6d6f8aaf4fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c4a69cadd3a653e18a75a5e05bfb80c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_489b10ce6faacde486c0c517f89b4dd1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e972c8efad4a7135d8637e299cadf6aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_489b10ce6faacde486c0c517f89b4dd1
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cb580bcd31a99e97b06004a6a86da029(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_963af844c566ee9f545825b6bb1bb630(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb580bcd31a99e97b06004a6a86da029
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb692de8a63004988cb6897cec3d708(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_974ca4aec87839abdc7818643f5c2125
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab87c7a3ed638bd876844dbc9a303930(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c3d47c17fade3bacc286be9f2b65086
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a2d5cf1b56d859e762466607827438c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac97e46c97fbf8e8c5b9c782718df1bd
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcfd0eca09ee719ca0744169dc20e70e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_080a58cebd644135cc1e10c9f224231a
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ba806fa61307c588082461886d8a14b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43fa6cdc4ac21f3103f7d33c72890199
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_987fee0a211015a2818b36ab268a958a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ec9846672d670e3ef5bbf0b181d44e7
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6854306fbdd7c2bde5088a53916f9ebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0163ca9e65e3b0c7dc77b6d5fbe9c193
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1ee5e3b1ad86dc487f5f1984873d314(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e35db49230b16435cdfdcf9dd0d8ede
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c51d8141c9879ee1a9a6edc5cf9ef14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0163ca9e65e3b0c7dc77b6d5fbe9c193
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46ab54553d7fccba6d586530452d2f4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e35db49230b16435cdfdcf9dd0d8ede
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9793a904dfc8937f0f4e95fd3b2ec007(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bda6c2b29a0d77efbca213c91271a8d
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79e1d4726567ae06215f3d71c4a44f22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_602cb6f83f46846b9ae579ca28c29757
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_587321ccf2c5b90fcbea9855d2cc74c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 37], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e852f78c74f4c5aa32e206f282598c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_587321ccf2c5b90fcbea9855d2cc74c2
        def get_inputs(self):
            return [
                paddle.uniform([10, 25, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 37], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25e1d6806717117bfb6cda264d3145bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67b8d4299af2db5931e65e6b59438f1c
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acbf9fa81f4e8331da7c66aaf665c9e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5cf1eb2d5badd705c4bbd8d69975dea2
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fe900615ed32af401395256415ed583(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[120, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd935d7ce3a701e90329571065c122d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fe900615ed32af401395256415ed583
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2577ab700ede2ae27a6fe4eb8777b95(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[30, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b70d6e56f22e9bf0ddedc98df877980e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2577ab700ede2ae27a6fe4eb8777b95
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.777217388153076, 8.48762035369873, 7.249322414398193, 7.724056243896484, 7.930503845214844, 7.815942287445068, 8.609721183776855, 8.345749855041504, 9.051900863647461, 8.07912826538086, 8.785380363464355, 8.09261417388916, 8.483768463134766, 7.842811107635498, 8.470643043518066, 9.749964714050293, 8.221514701843262, 7.603100299835205, 8.097103118896484, 8.626155853271484, 8.103931427001953, 7.6056227684021, 7.974206924438477, 8.232665061950684, 8.936210632324219, 8.685194969177246, 8.046462059020996, 7.636215686798096, 7.272360801696777, 8.230209350585938]], dtype='float32').reshape([1, 30]),
                paddle.uniform([30, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5db4813caeabb7540a88210d7662e63b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f36d755b39c6eb6c44503fcfd497b274(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db4813caeabb7540a88210d7662e63b
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b63400562f695a33bb31ff79c4e7529(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fa42a31329352c2a9f5d2cb0bee9bf8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8ef05d946429bbd7c4065aadd32ff0de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7c602c1ad719fa318dd9f4a402f0cfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef05d946429bbd7c4065aadd32ff0de
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_af9199acd390482ee8bf857828c7fe6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdcada177eac70b47552275bdc559792(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af9199acd390482ee8bf857828c7fe6f
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f36d755b39c6eb6c44503fcfd497b274(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5db4813caeabb7540a88210d7662e63b
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_78ec7a69e45c6f7afd23d453a1c068b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebcc9713069a95e2ab6dcb3922c254a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78ec7a69e45c6f7afd23d453a1c068b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b86a3141bd5561b15b137d205fa3973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef6fad89881b113f26334b36d3b9dabb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_94342bfad0289f6ed39a3f520e343fee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05b8092e4fc3232bae1044ad841c3cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94342bfad0289f6ed39a3f520e343fee
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f35d31bfbc55b18580995fd42a5bdde4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 5, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c2e0fca92e123988b67623447b8caf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f35d31bfbc55b18580995fd42a5bdde4
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebcc9713069a95e2ab6dcb3922c254a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78ec7a69e45c6f7afd23d453a1c068b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c59574e021f8c4dd6349e9baeff3f00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60d5535b91d4909aa070b4cfcaae835f
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6dcb5a90c4fe8adc2c44ca14a6d9818(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3ae86b4d3b7d237d1533ba8a8d51e11
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6cc5e02a3f5bcc30d7a1622513286cfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e16a99b0095e07e0185243d09bbdbf3
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3989d51e24c2d9a3742577b18fe0d0e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa293663fcc6a6d2a756b47de9443f0a
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6cc5733c34d475e7783abc0627e8501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_974ca4aec87839abdc7818643f5c2125
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdfb5e3a689e671f405d9d59ccba9370(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c4a69cadd3a653e18a75a5e05bfb80c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73b29c5141bfcefe737553b7f3ab84f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_489b10ce6faacde486c0c517f89b4dd1
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ba6d5ecb02441c7381ed8069167cb2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb580bcd31a99e97b06004a6a86da029
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6cc5733c34d475e7783abc0627e8501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_974ca4aec87839abdc7818643f5c2125
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_926b0d37508a0c7e989d13ba35de81a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f68c5d26ee0a68b510cc5fca64678ccf
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53a94552ec4b67c969a7ed8c58fa36ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bda6c2b29a0d77efbca213c91271a8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5df3c20cb42b9a7a2bf1371f92d5cee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8dac5f456994979340dbf976a4fe3fc
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4ae57cca9c696f0a380b56c3fb8737d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abbc7ad398f7168b69b6faaed2ffc26f
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67b903bd1f2c513dbcdfbd64871fc6fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e16a99b0095e07e0185243d09bbdbf3
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f560ee694e9a7612abe9fcacf64aa19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56e154929774af06bda3433d5aeea717
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28d322f65b8367b1779cb332b100055d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c11217327c69c7783553097801acc09b
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f39d9754a3941d4414aee88aeba207dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_260dcb5bc4f92b204818924953479c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f39d9754a3941d4414aee88aeba207dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29dbc1aee8e685a452ba9af7a4005169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0f3ecb1acf2e6f686c060250d08f64
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bcdba615dab5d9e5ac32a8ffa9f627b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4aaa412d30f41c148eb75fda441abc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bcdba615dab5d9e5ac32a8ffa9f627b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_63522a5069302df266185f8fe064f0cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_515faf420e034f3c82921ef6b27f9fd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_63522a5069302df266185f8fe064f0cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_260dcb5bc4f92b204818924953479c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f39d9754a3941d4414aee88aeba207dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_13016412264bec06775093c7e7f71521(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd8446a8001b60c81f9ecdb7770ae701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13016412264bec06775093c7e7f71521
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a9b8971db27df9375024cce64891048(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0f3ecb1acf2e6f686c060250d08f64
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_61b2168465db9b109aba1dd3a6d4a7c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bda591ed2de231a75ba91afb1c1bed6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_61b2168465db9b109aba1dd3a6d4a7c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_55047a8ec21e0665111a9556135051f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 2, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6894ddfd03c0b99ee5dced8981d3a1ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55047a8ec21e0665111a9556135051f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd8446a8001b60c81f9ecdb7770ae701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13016412264bec06775093c7e7f71521
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_112e9d6ed788f724b03f67c016ba28ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f68c5d26ee0a68b510cc5fca64678ccf
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b239447400c2cfb688b9146dfdc10a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da3ab7333c101aa3a0437bb99d8cec6e
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0496cfbc9efafdf580e5b5947afc47e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f22e51c8402c54e57dc6f22891f055
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ea5c2d3f4c15d6c44be12b5ab28325f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96e2f107b83b1032428957b0ad605134
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a784f19371c7f32825646c78a9fc779(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a7439a444332f16fc61953888234295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a784f19371c7f32825646c78a9fc779
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ed4ed711b9bb07137e719a2f7cbf4f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4367ee347a6599aeedf21da828a7ab7
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5905cee335700cdcb1c342807082e34(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 32, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c69eed3917442d1043f1897a0a49c15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5905cee335700cdcb1c342807082e34
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e525c2977b215d0e5610da0f08bae2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, None, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3416192340f8d0dabbbb95c1a2f32240(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e525c2977b215d0e5610da0f08bae2a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a7439a444332f16fc61953888234295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a784f19371c7f32825646c78a9fc779
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1c70000f2e392ab73335d43621c8225b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8fd6f3069167b3a05c975fcea88166b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c70000f2e392ab73335d43621c8225b
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9915f019fcd54dde3014e18636d93e1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03b6ef95c82ac7e22d53e37032e16780
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42d9921250bcd2294c6ab57822d01bdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_097a882373cd37cec13c19223ac87bd8
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f16956b58373ec8c41cacf5c9db324a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69bb3ad8a418a98695c6a4535d4ad111
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_172f7549c7f1e2daeb0c63cc62cd5241(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c11217327c69c7783553097801acc09b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23e7bed660a1ae1ff7e3c23455d5fb43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ef395254f3623bf4353c14b1f23efbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_579167266120051a30522ef5a7201333(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95d86f06c7141c58553c42c908a9db79
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_428b43737f64427b07900b3f5caac5c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_884b2c3cd6c15e07cae6e3ad52287c16
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a7f695fd5e1423be32e59000c59fbb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e89383b3679da523023ff28957d1d8c9
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9711096c0c271605e22aa50212a18d65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0dce33533d9eaf7ae6e97c493b4fe7aa
        def get_inputs(self):
            return [
                paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43d76e5e44d3aa81dac06e83473402f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9b6717e515acead502c5f2dd7f62214
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c05d51936b494dfdc6e84a6d98949255(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc71f981778a49f29ada0529668d2d58
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17c3aef07af8651cf37d20a55406f5c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30a9c9205612f91fedb5e035791dae72
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccfa2f7843256dfc279947599aab66c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc5cd4f845f036f61093a2165184ccbd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1803762435b7bcb04c8249351427a5ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb826489a79fb17d5f937f9c001cdb53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1803762435b7bcb04c8249351427a5ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29dbc1aee8e685a452ba9af7a4005169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f0f3ecb1acf2e6f686c060250d08f64
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_67f473fbacefb0ee4c7fb103999ca729(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6380f39ffeb04a7a1fb78f51f10304ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67f473fbacefb0ee4c7fb103999ca729
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c2f4f0f423acbded1864391617ce4972(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 1, None, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a79846b513ca727f6ff19867fb97535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2f4f0f423acbded1864391617ce4972
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb826489a79fb17d5f937f9c001cdb53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1803762435b7bcb04c8249351427a5ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5763e1c50b367dfe7b4c1c0b3a88a5d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[624, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_196a9bbdb523ae7f73376dd11e64a220(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5763e1c50b367dfe7b4c1c0b3a88a5d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.uniform([624, 156], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba9c82ecfab4dabbbeaaf803665d377c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
                paddle.static.InputSpec(shape=[156, 624], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c41e138a9be1846701b8d798ddfea2fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba9c82ecfab4dabbbeaaf803665d377c
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6cd652297bf2c0f6b51300d10c542575(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c70000f2e392ab73335d43621c8225b
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e518c7bd363dae315e17a88473204cdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03b6ef95c82ac7e22d53e37032e16780
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7ead7ffce8cf6af257e82374aef3d77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_097a882373cd37cec13c19223ac87bd8
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e6414a5c44be0d847793bdbfc2d606(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69bb3ad8a418a98695c6a4535d4ad111
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe14e500843b4f8010c6df3003dd217c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc9d025a23586c310e6748e79d7d1e29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe14e500843b4f8010c6df3003dd217c
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
                paddle.uniform([91], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_349596439a2452289ba9eb8c71b39572(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_297ddfa470981fdb91d2158f4eb2937f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0f409e328348101639473a4b16207f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ba385ab4a07866b5110ff83968ced56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12be8fd3903a93f0bfa5d230988105fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dada00bfd80fe93e61a34063450e49a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.to_tensor([[4.512653350830078, 5.044649124145508, 4.655038356781006, 4.622213363647461, 4.632609844207764, 3.9768052101135254, 5.023960113525391, 4.7768354415893555, 5.421357154846191, 4.686634063720703, 5.148271083831787, 3.8738279342651367, 5.151643753051758, 4.582184314727783, 4.700376987457275, 4.343968868255615, 4.77421760559082, 4.917938232421875]], dtype='float32').reshape([1, 18]),
                paddle.uniform([18, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5403ace578291cea59023b3fda4b7126(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.matmul(input_0, input_1, transpose_x=False, transpose_y=False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7492f04b665b98c21f2f173a1aef89f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb51d0024bf2c5c80a16185340c8ca38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18a6372142be23d9ea9e9fb7c78ec549(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65844b779b23e6861cf9a8d088d172bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
                paddle.uniform([92, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad9e5928d998896b1e3a15b7e3bcae38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.to_tensor([[6.194968223571777, 5.21784782409668, 6.012714385986328, 4.90752649307251, 5.907154083251953, 5.492164611816406, 5.724031448364258, 5.579870223999023, 5.25261116027832, 5.262575149536133, 4.954087257385254, 5.3679680824279785, 4.764451503753662, 5.245046615600586, 6.278081893920898, 5.532181262969971, 6.717957496643066, 5.762195110321045, 5.715753078460693, 5.413527965545654, 4.952664375305176, 5.497104167938232, 5.146821022033691]], dtype='float32').reshape([1, 23]),
                paddle.uniform([23, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8596a955b244388b91267dca6855cff7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f593fb89523cdabf8052e8a135a558d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2519b85afc7f95b8cfc1070d578e396(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed2ed1ad779a168af202fffca842f6ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f1d3e41e2bdc26ed263709d68c7e335(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e097d4c453ea7a8b80c4f7a06d40d9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1960, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3ab0358b8eb68b321d47916da1c9bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_180ace33352b682736d55a68aaa95397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_358aa699af1879d1b72ddbddff9dc4b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40c81df2a94945ad399989132242b309(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([960, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f977fab0265b13fd7c5eeff7f980108(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a318e44478df0b8c13685417069747d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_229ac9ab2ba2b77a27b7c14500bd70c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2eded0df43d06a8e33cafc6fc2164806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([512, 12544], dtype='float32', min=0, max=0.5),
                paddle.uniform([12544, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba75dc2f67f5ba07947eaee387b1fa2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd652835335eaeff0096e37bbfc7b913(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_739cb8605c0846743dc6b1b68c57f0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70cb2ff2939cd87e7a7eaeed51f6c01e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2384e09dc626b691841a57083608537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0010e9aa5c03687cb880c3b7e2c962c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f98f0f125edc98952ca238a933062b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c421c34a726189b6f289d6c215b3e6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9362b9b14ac844d8fdfe850e697516d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e23a80c3895a3a1df398ca7d8a22871c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef3bb66485648e7c007579156caab2f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cb0eb5e9536f2da3a419e8575a66558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e23a80c3895a3a1df398ca7d8a22871c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef3bb66485648e7c007579156caab2f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c392f85a2ccaa56176b202f8ed342a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af968e202b1a4ed44099469e19cdc0f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdf0fec914443eeda60d4fe193f27e57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62955a74821e86ad7c0e5f18d335ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8ef1206b1a358d463abd036c0c718ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26fa46421fbf42365307c8fb9dce9e71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38aabf8ce84ec9d3537ddead355740ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e58519e97e99c0ea77215d6ee2f687ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb09e5921f1831899c2ed5ee4d8fcccd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_261cc95fac8df6fcb91edf0a445f90c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ee5cc2e157ee0ceb09ef3e0c44ddabc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 577], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e18371ca10e0243824c6bc7614b2d56e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 577, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6312a0f165d38cca9c3be314f80f316a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_075909a6046f67c4ae8c4cf53cd02838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f47f673179910cbdce7698c824cd18a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_023c981009d2ff7c810e63b87fdd7467(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60f1a4415f7f70e18ae6f0b57ef4943e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1306ad27276a070ced7a3a7155f53ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e853944ea4ef282bbb20770b99852653(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3033718b31bec60efb3f6bb8ef2ffd51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cb5c55efb08641c0725282d73384e6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50fcc871a4faa11c5d9a3b4874db26f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_318f1dce17ec641268dc15d76f5dfc58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fbfcc166292ebf41ee39df887505c5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cb5c55efb08641c0725282d73384e6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c392f85a2ccaa56176b202f8ed342a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af968e202b1a4ed44099469e19cdc0f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da8a62c7500f50b8222cb80cf300244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab8a6c6ac656db98f7d9125f904959f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_833337cf68f349454550690e6afb4b82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([390, 3136], dtype='float32', min=0, max=0.5),
                paddle.uniform([3136, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e18aff84356d602ac0d6cbb6bab04147(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([390, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6779e197b69e3b2096ffe6b8a326d4e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bbc8215750685da286db2ee876de806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42fdf0e8dae40c5bcceed093637b5a5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f15086651a173197d9ddd1831883c3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4335824a44b0523fd7725333b6002954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 640, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e566b5ad65361e3c049c95954bac9f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0d72a2f6b4c619100347155ffc1b315(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd93c262970672da3586cc517e80679(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_131694e9bd29870b411c638e27595ce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c67ca6579c423fd9deca2cfb7b46ca5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f65a55b7cc09b95a80aa2c509d657732(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 198, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb2d9993f37b5f54aeb5224b2ef84fdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81b62321542bc8ad5658ee4d818d35b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c405b8d3f56bad5c788f2b9f320f3cf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5924973972014d5601ebb849bf982ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5924973972014d5601ebb849bf982ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([86, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f97b7d1217bcdadc271d86f750b9fab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0764d5e674064b83b7a43b143e2c1af3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0764d5e674064b83b7a43b143e2c1af3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([54, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db4f952ea083dfa1798706b6a1c0243b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c3d58642dec3f30cfc45b2c1ba661cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c49f9436d6f95e77293029e6efbc7bbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d347bbcba70bcaae24b2f5b67512be5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5648355c100cf1b60f538290753dc907(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([4312, 16, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd652835335eaeff0096e37bbfc7b913(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_739cb8605c0846743dc6b1b68c57f0f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70cb2ff2939cd87e7a7eaeed51f6c01e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2da98d5ba083704ae6cd120a3093383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a15fed3e7054cb78399a4fabbf36c0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee54a029700113a005a8e2d93029f9ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db66fe5b0faa0ec6e78d5c3690c7a643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4285186555f878c0248d4e099d185f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38a467a6cc08ed3dba1069b18bd2588a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61d310a66afced8c172031a11e2fb045(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db66fe5b0faa0ec6e78d5c3690c7a643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_965a251bb6026b78c8a503a96c9409f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4fa74431af18d222960035bcf17f93c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d7716428daaf518e3ed53dfc7a06f2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5403ace578291cea59023b3fda4b7126(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7492f04b665b98c21f2f173a1aef89f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb51d0024bf2c5c80a16185340c8ca38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 100, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18a6372142be23d9ea9e9fb7c78ec549(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bb836ade107bbc94c8f5b495fcc9a9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7a242c049e5d7723630c7d9baf274d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb39790dcb1a96c0b506778fd8188bb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c0b718b5a3d09f8dc6eb257fb7f8159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25b890493d8e18ce36690721f9a40835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bea67f53872a0a540b6e744dc393ad81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ca186155c5ebc192a517ce3e4be8781(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6779e197b69e3b2096ffe6b8a326d4e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bbc8215750685da286db2ee876de806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_080e4d1925bd27554bc33819bf051a87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e05bc81b706974848ecc792721f4e363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a71e6c6b01618cb97259e5e2db44107a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d30457cca5b958c3e5b440960f04875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8dfb05761cf6273833844ee4b823516(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a054d3b40210bed74ace45791ba1638f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
                paddle.uniform([1536, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3033718b31bec60efb3f6bb8ef2ffd51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3ab0358b8eb68b321d47916da1c9bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4257463f9d7b0615cf0fa53adfc6239e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f247fd4d3f2993efeeb2972f07ae61b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c460a7cc3ac77a8f9bfdbba077bea697(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_819b004e1060670c2a0e68c71dc7fe47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_430d4debbd0cb808289029721b19cec2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0973f6ae9f1e07aabe8ec93dd53c17b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59233393efe247d81d2117213707f998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_811c1e3abc0526733883f05943e3d100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49ade59aa500c850dbaf2ca64cf5bfef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_789d10b1899f8e6ff45bab105590ccc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_995f6e364e5b5387d850b5a2b2b86a24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([60, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab247caedc9c4e4e349a3e91d2a616b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6737b851b839d6f9caa0d61abf11bd36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4296b08423e4ddd95b629c41e86093a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98757e4d35a76a13b90a3e49dc1cd20f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_213278a039aefb5213e2de123675f98d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a367e57508fc223378700ab87a690da4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20679c86c8c23315f5fba5c39a2f2cee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98757e4d35a76a13b90a3e49dc1cd20f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf72bb1fe038ed9db47649a5f4836588(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8ffd1d396d93016badb02a726c1408e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9eef02a39b997be645fe6c66022ca99e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8567f0b875ea9759606c530f6118ddc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf72bb1fe038ed9db47649a5f4836588(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c0b718b5a3d09f8dc6eb257fb7f8159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
                paddle.uniform([336, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25b890493d8e18ce36690721f9a40835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_297ddfa470981fdb91d2158f4eb2937f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0f409e328348101639473a4b16207f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dbb8f7a7256802af3584fc7739e43e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3ab0358b8eb68b321d47916da1c9bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048], dtype='float32', min=0, max=0.5),
                paddle.uniform([2048, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_959b0e2c8b07a27c034b0635b3dfbc28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c49f9436d6f95e77293029e6efbc7bbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adace4587cd595d5ebc18030e742295a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ade9a880eed6820075e1cd5843f93cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24d48fef6db928e351086479674412ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([54, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6002d3b3032db09a434eaf55561d065f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64af42b102aa5527946b0037cce8aad1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da802217bac049376200ff2a7f66b502(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea8281fb28ea67c58497e2119aedde12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2dfbbabef347bbedfa340296ac75fd96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64af42b102aa5527946b0037cce8aad1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7502f92a4aa7df6f5af680919792b242(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb09e5921f1831899c2ed5ee4d8fcccd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 4, 7, 7, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81b62321542bc8ad5658ee4d818d35b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c405b8d3f56bad5c788f2b9f320f3cf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d6b699d08eb578d21b0799296b8fb58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 40, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 6625], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdf0fec914443eeda60d4fe193f27e57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62955a74821e86ad7c0e5f18d335ad9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 32, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8ef1206b1a358d463abd036c0c718ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 320, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26fa46421fbf42365307c8fb9dce9e71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5b8c719723cb788d7b4a566906caa9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c136c302e53a1d8e50bb2ec329ef743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e763a171828a54c778ded9e48bc40e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_080e27c4ee50c096e67f4884ce7645f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5b8c719723cb788d7b4a566906caa9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59233393efe247d81d2117213707f998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_811c1e3abc0526733883f05943e3d100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c76f22e2112978cc7de488f187f1c2ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ed6875c3e86731c41b54448ad09d5e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_681be16cf3a050c7d90a42faf651b91c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 32, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90b3c330eb17dc598f2b3d25d4e6b513(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 200, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_765ab55d0a1d95a7e26a83734a0bc912(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1435ffc5c4f4468918f0f0fba44234f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf8340b009ba38a2bd6b2b40276738c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d64ed6e300bd29b092a366798b3b3257(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([22, 197, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9a413c39103c0be485bbdab91e21e16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36b7be49aa8338f898846cbf33801a14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_051ebebe984bbf554c20b94b5e0a4d7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8af8eb7ff97f22ec9efa333d594a62dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_364daf53c274c931581320f60db05990(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36b7be49aa8338f898846cbf33801a14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_362e3f61aa9a37914eff4bc7ad77ecb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efe8e99a7f9fe2c0a34a4a57e1dc71ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42a76e62a6995840a9b8495997e72d0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5228dc3680368d255873e99089512ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4572ce2950e75531d504a9382644e142(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efe8e99a7f9fe2c0a34a4a57e1dc71ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_929ffa83fc877bb5b90422eace299927(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
                paddle.uniform([1248, 312], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d804e2eb7cc1fa6fd15c87a57694b624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 312], dtype='float32', min=0, max=0.5),
                paddle.uniform([312, 1248], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8372e71f3ce6d1116da9a7e76d4ca65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cee4e128077de64aa39219763316d404(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39a9db017d58bca871286ee4af1b26fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58956835231b09d3dec08c5d0f251298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86c060ece4bb4c711eaad654a388719f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7645a2e497bafe872da764a03911a436(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdf0f9afba8559fa6e4b25a87cfc1f7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7645a2e497bafe872da764a03911a436(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([11, 1, 1, 7, 7, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47157fc54a534dcb4f5e194319842d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cdc29b216c1031f47d14f1ccd60528f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b780041cb0d45699c294970f9c33cf08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1025, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16c40584dcb6c030348731ec2cac82f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdf0f9afba8559fa6e4b25a87cfc1f7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([11, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8daad9c8b1703501c4f31fe2cc75bd86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 39], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16a9873abf9848cbe1dbead27bf07842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 39], dtype='float32', min=0, max=0.5),
                paddle.uniform([39, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8659122e8193dbb444a474ca5537b451(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cd1577c4a7b6a010af7d871a52cbb8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf0e94f07f726b0a32ed50b44567db92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd9f31e79a970c3a519f5e4b07978511(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8659122e8193dbb444a474ca5537b451(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1306ad27276a070ced7a3a7155f53ad5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
                paddle.uniform([872, 218], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e853944ea4ef282bbb20770b99852653(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 218], dtype='float32', min=0, max=0.5),
                paddle.uniform([218, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ba385ab4a07866b5110ff83968ced56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 8, 7, 7, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9a413c39103c0be485bbdab91e21e16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce7a1f2aa2f51b775e82cbec0f8c7c50
        def get_inputs(self):
            return [
                paddle.uniform([43, 2, 2, 7, 7, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aea165749d9679f17e84958974f3837f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1a7335134e9c74431a7670f19f37bf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98bb20df0a901ea001e78c154f80746b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([22, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f67508f121af1ff5046ef30616567e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
                paddle.uniform([480, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e9429b1bc32e6f7c3358b3bc7f4dc79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([145, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da8a62c7500f50b8222cb80cf300244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab8a6c6ac656db98f7d9125f904959f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c3719ba2ddd3c54596dec4c5e8c8e06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 25, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 37], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c47f4a48b6edf2597cfb0f1e13689b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3071fb55297e3bf14e7d58be71d49bd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([171, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([9, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d60952856c00f79809d02d6506b85bf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([120, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b43d2a15ea2b6623e4e3b5812db62f1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.to_tensor([[7.777217388153076, 8.48762035369873, 7.249322414398193, 7.724056243896484, 7.930503845214844, 7.815942287445068, 8.609721183776855, 8.345749855041504, 9.051900863647461, 8.07912826538086, 8.785380363464355, 8.09261417388916, 8.483768463134766, 7.842811107635498, 8.470643043518066, 9.749964714050293, 8.221514701843262, 7.603100299835205, 8.097103118896484, 8.626155853271484, 8.103931427001953, 7.6056227684021, 7.974206924438477, 8.232665061950684, 8.936210632324219, 8.685194969177246, 8.046462059020996, 7.636215686798096, 7.272360801696777, 8.230209350585938]], dtype='float32').reshape([1, 30]),
                paddle.uniform([30, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff0f2863ffea063caf6d9fe4f32c838b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb0256b89b5303add475ceef0ede33de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73c771ecff9303099280ccddd8496859(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25b6d86b77a631700c863b52d973840e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff0f2863ffea063caf6d9fe4f32c838b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46ad05c36d45e8e97892697c86806af8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74829c67cac42c49a3edf14761a908ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f90e04a0da0a9e680b0a488351eb22d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3afb405261dc6f787431b4e55ba5954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46ad05c36d45e8e97892697c86806af8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6737b851b839d6f9caa0d61abf11bd36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
                paddle.uniform([672, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4296b08423e4ddd95b629c41e86093a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([168, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0010e9aa5c03687cb880c3b7e2c962c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f98f0f125edc98952ca238a933062b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a62b710f471506fae35130544e19a4a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84e72223587385537c5ee3f0f92ebf34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a72eb1c1aadf361eb85d0f10e3198683(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 64, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fad9c15230e4ba82c0db88d7a6b49d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8, 512, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a62b710f471506fae35130544e19a4a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b375e955f40a933e623eeb87e9b51ffa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb856ccb74aa0ef6e389c34a1430ac05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_180ace33352b682736d55a68aaa95397(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_358aa699af1879d1b72ddbddff9dc4b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4fa74431af18d222960035bcf17f93c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d7716428daaf518e3ed53dfc7a06f2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4ff3170656033c1571994cec9e73e94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d420345d56fcad1f0111cc78a0a4f393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d30addf9ca7ca36146eb76365bfba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed076992015a22f98fdf97297f0270e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_605039a1c9afccf2fb4e70a28e44296f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 1024, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d420345d56fcad1f0111cc78a0a4f393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d4d747d584950e9b6f0972f522c1f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c136c302e53a1d8e50bb2ec329ef743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d3127fc7c1d19b944a8fa2121818e19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_856f8825b15931b8fb8dd37c85f5d1b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d4d747d584950e9b6f0972f522c1f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fee5ac117d069403ade3991766adccb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38a61f46bd686f32793f06274800931a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 64, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efb4037e836b242a6a5dce16534d46db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
                paddle.uniform([86, 3, 197, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b2b0b5c56030d6024613d17516aa1ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_607a2215e86082124fa4c5f2e21ae96a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf7cbd4032c4c965755ad7a88c2cb25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36d33dec13b74026d89dd2e7eec30f9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fdb3dd1b63abd91b9c8e4597fde6d4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 512, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_607a2215e86082124fa4c5f2e21ae96a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fc1f020f25194b31e63ba8b48d7a082(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77a6310d7c9b8dcb41772988a359ae43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8205e00fb209867fb632090f1f914864(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 160, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df7bbff449eddff8152cce753391c7b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d3b3f908a760dde6861413ac2b848b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8458b6f0d6768e859c8dcc1ca99a942f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0d34bcab9b995b985cf9fdf7b249983(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5f963e90e3d818af7097ab36ffce162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b6c0ddcc22cf7389a82eb7b9fcda52b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12a8b11c6662e29ecb391a2a764abb0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
                paddle.uniform([704, 1000], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b909e0f2fbb06c14bd10e0201c29219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7d8b60ba71d5422eac965ce25ed9808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 64, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f202df7045ac3e0b460e66fe108a61f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12, 1174, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7929cd352a4e53790e4ea765a8e5470(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5532c8b4dab329a82282db5ac120947c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d30addf9ca7ca36146eb76365bfba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcf9ff6dd6b460108d7e2292f003e848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a9904b063da88b2ece3df561b57c712(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 1024, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5532c8b4dab329a82282db5ac120947c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05fadd34cdc9ceeab14079c3c54f579f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
                paddle.uniform([624, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95bc5ad3dfc671124e0b6831261f8e4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f3990b4e0b90475b9b38e6dd2c3774e
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
                paddle.uniform([156, 624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afbe3da3e83aecf2cc45f88a8a7c7f91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0faff1cb5c54f3ee8c35534c8da9ee85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 32, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5515ea09ff2086c5cd11e5b86fd17273(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5930fa19a6e6e58bf590612b7ccca3ce
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 50, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efc80e17dd03d943ea588e7e969407d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_349596439a2452289ba9eb8c71b39572
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 256], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()