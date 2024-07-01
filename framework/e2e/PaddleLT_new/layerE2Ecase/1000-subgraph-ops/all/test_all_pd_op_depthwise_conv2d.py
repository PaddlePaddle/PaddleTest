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
    class PrimitiveOp_f00a192436fa1eadef6e052b4a87a0ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ee4886f52e6efabe5f56a33a6564ce7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f00a192436fa1eadef6e052b4a87a0ca
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0c0ad4c10b3f117431252104792c8c93(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1248ecdfb920a115a8caed6f41199ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c0ad4c10b3f117431252104792c8c93
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_56e8c467125dde22ae125289eb6c3c60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2203698d8f32fa2cec0cdb44ac525df5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56e8c467125dde22ae125289eb6c3c60
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_04fc656b09fa49576f2f0069caceeb05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83227e14cfab69d111a172b8b3c2ea0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04fc656b09fa49576f2f0069caceeb05
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83227e14cfab69d111a172b8b3c2ea0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04fc656b09fa49576f2f0069caceeb05
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c8e3c0ba04d47b7e58af1a5da1a16a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a85fd2d3ac538eff403bec765375b96b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c8e3c0ba04d47b7e58af1a5da1a16a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19ae726ad41380721c10c9757612ab8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[128, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d08aaa6e57f9502742d42d56abae49a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19ae726ad41380721c10c9757612ab8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ecde47beb8d016d84c1eec93df94d13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 1280, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1280, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1280, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d41102e291088430ca51c07e3852472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ecde47beb8d016d84c1eec93df94d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_10df4bf13c9ffdfe946390ad8eeeb952(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8a3048deeb0ab279e0a14c57460880e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_10df4bf13c9ffdfe946390ad8eeeb952
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57ca5d4608aee1c9998426b661e3cdf2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa7ac68f47148620c7fc188a0e5527df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57ca5d4608aee1c9998426b661e3cdf2
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f7157d48fa92441c80ba34b036996a56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_354d76de0ca3c3c85481f6ec9508c6e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7157d48fa92441c80ba34b036996a56
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ca6f58fc8ee4d57ab549b291bc4f1b34(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad7b81a628ae14f9aba21b0df87d663b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ca6f58fc8ee4d57ab549b291bc4f1b34
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fdb1023f930066831e23c6df4b50d1ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ec406147de82f2726e7859887981fe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdb1023f930066831e23c6df4b50d1ca
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87b9928358b2ef12630bb12df317b356(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9578e197794056faab8a8173ded8c9ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87b9928358b2ef12630bb12df317b356
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fada6d9bcf11d2a8dab037d0b17b1cda(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[160, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c2c895f764200b8522b9855a62decb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fada6d9bcf11d2a8dab037d0b17b1cda
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ab621b2b22f5b9683fc992183e544d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 192, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f709c5f0f388bef409730d32bc3e0d44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ab621b2b22f5b9683fc992183e544d8
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8ebe92c47cc0858f74aa87f3526072f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 384, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30478e69fda4e233f8a4f498e7cb84f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8ebe92c47cc0858f74aa87f3526072f
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c8b48381bd9bd6932c57d8a6a3d775a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52db23cc713c508bdd75c2d1b7630df5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8b48381bd9bd6932c57d8a6a3d775a9
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_705e24b516a7b712d3758703f3dfcf4f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_802b2289971b893fa256684d6c0cc0cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_705e24b516a7b712d3758703f3dfcf4f
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c72c81285fc89c690cd73231bbc7c93(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_826b200389c82cf10f097bfab69ccf2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c72c81285fc89c690cd73231bbc7c93
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba446018c6f98e6ab039ee50ae08fc71(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c73dd04203852e5502cb0134fda9a725(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba446018c6f98e6ab039ee50ae08fc71
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df9fb737c2f725b51933ac425475c9b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_612c364e3e1f22560c7564b3348bbd05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df9fb737c2f725b51933ac425475c9b9
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_081a8941e17c209311a797bf683b107e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_062178e421e713f649e7420ff680bef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_081a8941e17c209311a797bf683b107e
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f46e3645b6172e272c4898fa707aceb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 1, 9, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d9152e2135427fe8c28510c5fa1b2d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f46e3645b6172e272c4898fa707aceb
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45de591735cedbb5387f8fc679cbc6f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f0c37ee3220e33d20873e6c877c24bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45de591735cedbb5387f8fc679cbc6f9
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_77d6eaff22588658a042af9a0b7a6758(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce7b82abd6fb2e00103f56e22965719d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77d6eaff22588658a042af9a0b7a6758
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49ce42732f3c09ecc9e2427064583483(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b9764b97ab2a714c54dc20112483fce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49ce42732f3c09ecc9e2427064583483
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_add6248edb8c06b2ca63507eba629e06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d924d45d54494e24a351ace3b3271691(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_add6248edb8c06b2ca63507eba629e06
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9acf89b94f49159e7087e491f10bdcf7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[90, 1, 9, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2270b51c09b9a5dbf81581efd883342d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9acf89b94f49159e7087e491f10bdcf7
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fb43006a4e9ad58767a31e01b7c4b84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 384, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[384, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5be1af4c20fa3e25eb0be42dee3209e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fb43006a4e9ad58767a31e01b7c4b84
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_007dff9dde2c92964bb3bfccf92ace0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06912044f1de142b251b9a499f5a7cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_007dff9dde2c92964bb3bfccf92ace0e
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b76d5ceb58808bd2ddc4b7aa73313edd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68d3bd88c69b8c0591140d6daca92b71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b76d5ceb58808bd2ddc4b7aa73313edd
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f4a371cac94cbd8189b13c60199d14ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f9f52d1b917d2eae0c43b8c1aa3c0a3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4a371cac94cbd8189b13c60199d14ff
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_610c4264731bcf43b76303c71d192fdf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [4, 4], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 9, 9], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32b90db6b2919d697a787853c5c8c414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_610c4264731bcf43b76303c71d192fdf
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8f4e858f18cab351e3c7c5561f574ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [5, 5], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[144, 1, 11, 11], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60864ba04ebe99ff2b57b17a4a19d9a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8f4e858f18cab351e3c7c5561f574ea
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_428cac4300acd1937b0fe15be0d50e0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 768, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9cb4699a7b647bc8705df8000a623709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_428cac4300acd1937b0fe15be0d50e0b
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_154db11a41e3d3977aada5829fc4ac78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ecde47beb8d016d84c1eec93df94d13
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2278fe4ec753087799db559de6fd8bea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 768, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[768, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_058a6ed4653a871d33313db5fc507dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2278fe4ec753087799db559de6fd8bea
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df3f0f073d20c1807d76e3e47710a3f3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b6ae10803f0ec26cf4486b95ab40dff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df3f0f073d20c1807d76e3e47710a3f3
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d144d054987492ed0d1efce229242763(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c87fddfba25d306316f49615fa4e9715(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d144d054987492ed0d1efce229242763
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_715c632e0f279de45d618fd6ad46fbc5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80806ef69dd279d404dd3ba474fb02a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_715c632e0f279de45d618fd6ad46fbc5
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4833875c15733cce7885eb1b40373f93(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a87916f59fc0abf3e69c83aee43727f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4833875c15733cce7885eb1b40373f93
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_66a62c84af25ea4773b1c5abf865ab3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_355cf122694d6adba36e9e33573c1089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66a62c84af25ea4773b1c5abf865ab3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2af7380f67da80d6b49a37b60c886e30(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d83f34d244454eaeb869d46cdaf7529(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2af7380f67da80d6b49a37b60c886e30
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e630928dadcf1f10545729a718116eb1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3dad398a4406f66d6029107228e446e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e630928dadcf1f10545729a718116eb1
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a43141ba250f2778cb90d53e2b5d016f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5a69a168236965e43d5e61452576ac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a43141ba250f2778cb90d53e2b5d016f
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be58543b5e3986bf4eb4d8f05bdf27b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 192, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[192, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01beb797de5705d1c23b6102d251729c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be58543b5e3986bf4eb4d8f05bdf27b3
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8b572092ec71a23d7488d709dd9d4e0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_062f0a7e49a685e87f511cd259c44395(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b572092ec71a23d7488d709dd9d4e0c
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a5124a76f59be2926d561b65e5e8fb58(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[240, 1, 5, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71884fc529c312edd7b04daaa6d0681e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5124a76f59be2926d561b65e5e8fb58
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e1a6ea3fb6498aed34b8e8c2887ca41c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 1, 3, 3], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae0fdf0285ba59980457b94e714740c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1a6ea3fb6498aed34b8e8c2887ca41c
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ab1f3a63d6d4c59e6edf82f0450423b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82aa5ba460c171971843dcec1b4c20f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab1f3a63d6d4c59e6edf82f0450423b6
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c6ef47e4dafa902e5d5545a5acf67e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_992c7fcf1c3833426fd32af71fd76ffe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c6ef47e4dafa902e5d5545a5acf67e5
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_42359b7662b798539344fd0b9afa1df9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 48, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2351e99ba335dd00adfe424d5f137943(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42359b7662b798539344fd0b9afa1df9
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7552eb498e8c56ea1116b1aa089950c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_155bf2d9c82a0d54cc4818870e6db42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7552eb498e8c56ea1116b1aa089950c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_155bf2d9c82a0d54cc4818870e6db42d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7552eb498e8c56ea1116b1aa089950c7
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_559bd6f5cb1a31ef323fef62788ee617(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8747a8893696a95f730c947ee4636ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_559bd6f5cb1a31ef323fef62788ee617
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_392f1f75b865da9d0ee78b5363964205(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 128, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_faa6bc2f107bc2cd18754705bd47ff46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392f1f75b865da9d0ee78b5363964205
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_70d703d834560f4a81beea54f21590bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 1280, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_966618632249735ee413f6580ec72f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70d703d834560f4a81beea54f21590bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aed782920ed2140712339027582d63be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f9c5c2c8951a87f6b8008158b000e96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed782920ed2140712339027582d63be
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_536a142995e1695e0ce00c5a2ab0665c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d279de1018be9ed61b9612a8f9d6685(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_536a142995e1695e0ce00c5a2ab0665c
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0eff887a7ca97bb26b9620be7fa481c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77969f0c68d661babdde09f9ff772a3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0eff887a7ca97bb26b9620be7fa481c
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2f236a8fa3a153acd28df82b715f3ab2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [0, 0], 'SAME', 64, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8fc02a5da5b365de82d0ef689fd0d2fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f236a8fa3a153acd28df82b715f3ab2
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_776d642e8c599bdceebfa1a7ffc3fb0a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ddcc17e60e93c9302f85fc2405e5ec40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_776d642e8c599bdceebfa1a7ffc3fb0a
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_190a1d522c1b9f84c2853c6002c101f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a064f3b4482672390d0d0e865ca49dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_190a1d522c1b9f84c2853c6002c101f0
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0becaf7c27288304e8c50a31d9da44bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 160, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd949636558c894a1f8c68072a3f509e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0becaf7c27288304e8c50a31d9da44bb
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([160, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2de5848522313251b428bb693f2ea55a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 192, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d00267536a46b2937caf19497566ff3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2de5848522313251b428bb693f2ea55a
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_80e2db2e1510103cd630e65a92d2f8d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 384, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3015b27a3319e66aab8fed005b4f1ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80e2db2e1510103cd630e65a92d2f8d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b8cc52e5d3a574e483ce9981d029554b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24b8d57489b809c76ce087d946840cd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8cc52e5d3a574e483ce9981d029554b
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49efc4072075eb4c5892583ece11272f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_511def2d2963f25c243adca0571c1073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49efc4072075eb4c5892583ece11272f
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_26b274f5ef31d24985064ae6b1196809(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 80, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d58b9c054e00e9bb61d1c698764cf6ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26b274f5ef31d24985064ae6b1196809
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6ffc726083f2ce2a19b14ace8fddfe96(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_58708684c888ab18bbc8999a923ec368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ffc726083f2ce2a19b14ace8fddfe96
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b7670f5f3fe3329ae446379fbdd27b2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71b265ebbd4f6b29d2a8fe476e3c1b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7670f5f3fe3329ae446379fbdd27b2a
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_25eadd3899ac7f221f9b404c354c484c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_787ac4e27177e95bfaae68a42c4b5ff2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_25eadd3899ac7f221f9b404c354c484c
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8026c8f5dba9ff72e132e620092e3e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 300, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14d5d7eba2dd40da272fa51a442e0aea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8026c8f5dba9ff72e132e620092e3e8
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2db54be42a25610e2f394df567c03a47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 96, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_952f056ea77700dc9272ac2a30a3b525(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2db54be42a25610e2f394df567c03a47
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_67914abf755f416a55108d3f581df440(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8620d1fc76fa584aac7f7a0b1d8d9cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67914abf755f416a55108d3f581df440
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_acf2021039c6e9200e9f133593ccad2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2595d43119517090c65b9e23ae059d71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acf2021039c6e9200e9f133593ccad2a
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fdad4d561fd09a0a331e7222608df41b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [3, 3], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a4625a09e480d26aca09db4faaf67076(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdad4d561fd09a0a331e7222608df41b
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7f6d550167890b077882c127cec978e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [4, 4], 'EXPLICIT', 90, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_227a0abf537f403c6c45136dc6f5cbe8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f6d550167890b077882c127cec978e9
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([90, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_667c3ff5d640ce1ed5849a9782e3ba66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80e2db2e1510103cd630e65a92d2f8d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([384, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2fe55615ba4e00c516fe77bfee2d184f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [1, 1], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8800cf3a503992f541dbceec5772e58c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fe55615ba4e00c516fe77bfee2d184f
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_edfe799188f8ab2b0c4af3d11c5abb08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [2, 2], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_316500b99c3b3f0465a019436f3e96f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edfe799188f8ab2b0c4af3d11c5abb08
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa844f82ac071343860fdc7c71f5ce49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [3, 3], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be7955d6fc8adf9cd3f0375e9435ace1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa844f82ac071343860fdc7c71f5ce49
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce4441ea962800888d17864a415a484a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [4, 4], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b00a9d80c788854fba6ea0354ae5a236(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce4441ea962800888d17864a415a484a
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0e722f276b55d06d6d2b6ea08542ee3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [5, 5], 'EXPLICIT', 144, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_adafdf6253bcef2818412531087e46c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e722f276b55d06d6d2b6ea08542ee3
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([144, 1, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f762464138a333540ab9632c0571c7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 768, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e4244dab6fae94d9d7146140119f961d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f762464138a333540ab9632c0571c7c
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8db0b8a1262bb18cfb5ee4034ef9d369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70d703d834560f4a81beea54f21590bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1280, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00039fa40b35050252b4d72f040d2246(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f762464138a333540ab9632c0571c7c
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([768, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0433b5a8c614ffcacd006834f5ac9637(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fdba3ef86dd156fc656fbe24660c0afa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0433b5a8c614ffcacd006834f5ac9637
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a0a19b915245a640490025ac627df07b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e62762d409389f81c14457b6d2efa38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0a19b915245a640490025ac627df07b
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b0cecc372cba1ec29a4ae48a19b4d517(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_420c8420c52c43c81ec766ce66c19620(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0cecc372cba1ec29a4ae48a19b4d517
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eabcfa8ff1856bfa53aef88b641e9cb1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 24, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dcc0e836c1c30810ef532cd576fc2621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eabcfa8ff1856bfa53aef88b641e9cb1
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0c294f2bad408f3193a5a88a141b57df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f30160c229d29dd14fc6924ca3cf7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c294f2bad408f3193a5a88a141b57df
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3b2ce99b554d4a461527b85ffe9e0a19(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [2, 2], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92948e5f86a9d898a1a511aa6c3281a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b2ce99b554d4a461527b85ffe9e0a19
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dcd845163804b459033fed63170bc1a6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [3, 3], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82a019ee9214bbf563a71c36b1e3c522(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcd845163804b459033fed63170bc1a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a3f63b5e4fe7efc32a36a7a9ed2a382(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [2, 2], [0, 0], 'SAME', 32, [4, 4], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2940c0a98d2954d96a6f8c0d077d99d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a3f63b5e4fe7efc32a36a7a9ed2a382
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7147841fc11a1de3d172a1e3f7e60bb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2de5848522313251b428bb693f2ea55a
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([192, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c6779f06d0bcff205fdba068f4590937(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [1, 1], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72b102038a264f2245741fc4b7231ea2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6779f06d0bcff205fdba068f4590937
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0c22fd195b2ac9892316333bb5381fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.depthwise_conv2d(input_0, input_1, [1, 1], [2, 2], 'EXPLICIT', 240, [1, 1], 'NCHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_228a11379ffc20f5390fd2a1625ccbf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0c22fd195b2ac9892316333bb5381fa
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([240, 1, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24b0a18fb0b285d79f2596b30ff259cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2db54be42a25610e2f394df567c03a47
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 1, 3, 3], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()