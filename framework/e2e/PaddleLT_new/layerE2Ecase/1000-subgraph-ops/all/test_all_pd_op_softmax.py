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
    class PrimitiveOp_d8fb3c008afd23f13d0f8a38a2c77599(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21504, 1, 91], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5abe8fa811f54306ddba409457dab505(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8fb3c008afd23f13d0f8a38a2c77599
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fda971a456cb96a62b38c5a2c314b7a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e57ad18c39004d1c4b9d2a755ec9bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda971a456cb96a62b38c5a2c314b7a7
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c3f35d785235adf039dd0017c40186f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 198, 198], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5555e7df38beb953b6e7e098ab36f962(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c3f35d785235adf039dd0017c40186f
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_307ea528ad0489a42dc1c32d01fffe5a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 4, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_afa2cfcf0e84545194ecf7fcd8f299f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_307ea528ad0489a42dc1c32d01fffe5a
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_602ad46e69cfd76d114997999b584c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda971a456cb96a62b38c5a2c314b7a7
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c80a6685576db51c60930d6d57a59a75(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 19, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bfe2e6e6a352315733367148cbcad98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c80a6685576db51c60930d6d57a59a75
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 4, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b683fc137753fc502f1663095c97fdee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ff99f79e24cf94966865a24fb5c95ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 21, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc813dfd9917811d4754f7f21d27a411(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ff99f79e24cf94966865a24fb5c95ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2083bb8e266a8afa2aa17afe3b32e959(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c11303a790345a60456fccbbd4e2a58e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_206fcaadebed9ecafece28bb6cb4735a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c11303a790345a60456fccbbd4e2a58e
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c6dcbd50b9dd881009335bde6811c29(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 16384, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d54c49e03254962471606c0875ba8756(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c6dcbd50b9dd881009335bde6811c29
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7a859455dec842e1e4e7454760eedc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba25d29f149059b5aea6d120102ad36c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_720d310696f4a7b2aac2bcd0a9b7bac8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_711c616eeb88f73cb3fe938569f28088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_720d310696f4a7b2aac2bcd0a9b7bac8
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1c8d958f6461b3e6aa5f2a8e7abcff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c3f35d785235adf039dd0017c40186f
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_11ebec3f2ebfc8cdedc3b40e2db3a76e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af53d3ca54494d2e10ec99217bf23bc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11ebec3f2ebfc8cdedc3b40e2db3a76e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e57ad18c39004d1c4b9d2a755ec9bdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda971a456cb96a62b38c5a2c314b7a7
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca19de753bb92871ef6ad344a00ab443(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81c4ff3237c675bfdb4c3733f6c1803f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c08c48a70a804a7c9cb48b9a0ae4de2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd5e40ff9a3f09a64c8b375257a23421(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c08c48a70a804a7c9cb48b9a0ae4de2b
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b28978ce66eea4a2d3e11ca2d30cb328(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ad9e7d8bda3952c2586e5a45d2d3d25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b28978ce66eea4a2d3e11ca2d30cb328
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7d07b65ab8b05e2cb43b77fb02c4152e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e0de0ff7e98b9a2a53fbebd42786df1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d07b65ab8b05e2cb43b77fb02c4152e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ad0605051e3ebced0b30c016e1d245e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 2048, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_35d7a6a6219948f3cb2adbbe068d9e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad0605051e3ebced0b30c016e1d245e0
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09529613f2005a129f5d76973dad5871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11ebec3f2ebfc8cdedc3b40e2db3a76e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45a31c834fe1f281201c34e8aa853c01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, 197, 197], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1a1389f082e97c3be0439937d281c83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45a31c834fe1f281201c34e8aa853c01
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3464cbbdcd54e7a72e5d9635a889bad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 65536, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be4657ae1ee3a16004f679a6f59469dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3464cbbdcd54e7a72e5d9635a889bad
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_602ad46e69cfd76d114997999b584c73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fda971a456cb96a62b38c5a2c314b7a7
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d13ec29d62196bf47a0c309772073da0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32768, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_580dfab9123820ae332d1c11e762e2c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d13ec29d62196bf47a0c309772073da0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af600c2c9fdd8346b70bb74fd2ba0667(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_720d310696f4a7b2aac2bcd0a9b7bac8
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54d8e6b1b53b4e3a8b60c76dc16b0e2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_38907e12bf70476c14a653ef14e7d838(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2, 8192, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7ad0b3d2207e9d576726a728859c0e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38907e12bf70476c14a653ef14e7d838
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35d7a6a6219948f3cb2adbbe068d9e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad0605051e3ebced0b30c016e1d245e0
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cd5e1b85bee20745eed7c8a859cc8ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b34daadcb8d835abd4fa81b22b0501d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d07b65ab8b05e2cb43b77fb02c4152e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad9121e11e1014be2ff6d47501094dc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c11303a790345a60456fccbbd4e2a58e
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d452ae072ce8b0068eb54afe126c55be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5b8172d371465fd581ee3c882e1ae67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d452ae072ce8b0068eb54afe126c55be
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09529613f2005a129f5d76973dad5871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11ebec3f2ebfc8cdedc3b40e2db3a76e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b47002d7249731fefffbb3dab8e541ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2de051003cbc3c13bb877c7ed23645da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_839eb3870f21c2ade77ab2fa15e19fb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2de051003cbc3c13bb877c7ed23645da
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eaa1ed57de41fdaecb9e0788964e0b34(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5, 4096, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_acbe14027b4eb80285dbe9a34a70ac10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaa1ed57de41fdaecb9e0788964e0b34
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acbe14027b4eb80285dbe9a34a70ac10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaa1ed57de41fdaecb9e0788964e0b34
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af53d3ca54494d2e10ec99217bf23bc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11ebec3f2ebfc8cdedc3b40e2db3a76e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d54c49e03254962471606c0875ba8756(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c6dcbd50b9dd881009335bde6811c29
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ad0b3d2207e9d576726a728859c0e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38907e12bf70476c14a653ef14e7d838
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a39f62b0d139869c6fb56c26f53e840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45a31c834fe1f281201c34e8aa853c01
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_580dfab9123820ae332d1c11e762e2c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d13ec29d62196bf47a0c309772073da0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5b8172d371465fd581ee3c882e1ae67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d452ae072ce8b0068eb54afe126c55be
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c7913e95daf735a882be9ddf9373fd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d80a94077c3b04d6d79c06be521b6ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb44857c8e95d99156285e6055c36bb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11ebec3f2ebfc8cdedc3b40e2db3a76e
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_289c24a7c9462099fdd80327ec1520c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b28978ce66eea4a2d3e11ca2d30cb328
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5e649c3a5a03229b2b8e6d10fc0a3b79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78f402fc7686c5988492728b50654172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e649c3a5a03229b2b8e6d10fc0a3b79
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_497d660c0034f9ad93c85640d7689547(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c43279f7e75c10afb649eaa4c9009aeb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_497d660c0034f9ad93c85640d7689547
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14d1019fcf2791ec678097fa800d4424(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c11303a790345a60456fccbbd4e2a58e
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be4657ae1ee3a16004f679a6f59469dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3464cbbdcd54e7a72e5d9635a889bad
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb3c25a9be8bdc8430f37f38b865aaa0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, None, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d623c845f4f1eb9d6a6c9850adde267c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb3c25a9be8bdc8430f37f38b865aaa0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8a269766e5ae18220928249a649dce7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11ebec3f2ebfc8cdedc3b40e2db3a76e
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_275856d243390f990a99c2c14d2e951c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68e7dd1e48723f5db28304dd9380afea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 1, 91], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501223774de64cba2d41e15988dd7d90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da75fa26c4614a3452e4ca1f19d0819(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f8b7aa4b9484d5b70aceb78464700ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc06eceb9daa1fa787ab758fdc635ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be8b3570784955695786d768138a564a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b68ef159bced3390697a71bd66c6140(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8b3570784955695786d768138a564a
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1596bc0bad10ec716a8927046a664ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92c88ce6eb9d65b8c1958ace7bfeec15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8b3570784955695786d768138a564a
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2e0798e5b1b91d489365fcf13eb456d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f1cad70f8e660da3920ae4ec0c8774f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 577, 577], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_359911c6611a7653fb6a66980944c815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5ce5f334983080fe5f170d20dd09138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b88b5ad927bf0720b4e6f92a088a71a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdc32c6666f1a5e509d2c093b86c810d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 640, 640], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_774391f609f9383e729bbda8ebe45da7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 198, 198], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd13889791d6cc9d504f1965be73179e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501223774de64cba2d41e15988dd7d90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 100, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5be2a9726fd8fb148f60dafa20fb6dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d6cd44d701393f521cb5425c4dec61c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_92b43abe207d2d1598094754dd8435c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.softmax(input_0, 2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f1749eaf83eb0410ad2737640d3940e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92b43abe207d2d1598094754dd8435c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3579137e90989f55e2e2c3e1e8b55566(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0de0ff7e98b9a2a53fbebd42786df1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d07b65ab8b05e2cb43b77fb02c4152e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_859460cdf459aa84cea8c439cabd217a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd7a52b006e123a449fef5fcaeb78677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_892b3651784e01f4f66390bff5b38d2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([54, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10f768cc7c71c16573efa7a89fb4e441(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dc06eceb9daa1fa787ab758fdc635ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 320, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7a06983622aaed4f1dbb95f3d132879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_519c69f2dbd9e5053b6e1b15f5f6dedc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 200, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e44ecdcc8c65a528bc840d6196014d82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f6bfb19d99a5a5e56db266a68e0cb19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_859460cdf459aa84cea8c439cabd217a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 2048, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56943e1596fde89da7ae4f2deb3887fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b34daadcb8d835abd4fa81b22b0501d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d07b65ab8b05e2cb43b77fb02c4152e
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3c1e3de9043473ad6ae098d740e39bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1025, 1025], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd316c223af5829b56a0ffdc703d25b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d07b65ab8b05e2cb43b77fb02c4152e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd7a52b006e123a449fef5fcaeb78677(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4ce18b6f26533c9660736f1ae905945(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_090b9be3c23bcf25612e60d6966d5372(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92b43abe207d2d1598094754dd8435c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2107dea25bd4cf5ae248a2b391d11e99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2107dea25bd4cf5ae248a2b391d11e99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 4096, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd13889791d6cc9d504f1965be73179e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_359911c6611a7653fb6a66980944c815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16384, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f6bfb19d99a5a5e56db266a68e0cb19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2dc6f446711fd81a4c77f75329b58475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([86, 3, 197, 197], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7a06983622aaed4f1dbb95f3d132879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd316c223af5829b56a0ffdc703d25b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d07b65ab8b05e2cb43b77fb02c4152e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3293d183d548df455357aaf22f19152f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c1bd37a3dc1827200f04ed917615ea6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f5b527b44456d324a0308042d07185f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00ad63361b8715b4c05fe000b38add55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92b43abe207d2d1598094754dd8435c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a747c34df4b0ee341d58f720812d2b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92b43abe207d2d1598094754dd8435c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d9bdb8ccd72b46df371e8cb4804265(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 12, 1174, 1174], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10f768cc7c71c16573efa7a89fb4e441(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 65536, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2753ca6740e60ce77bcbfe06fd4c564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92b43abe207d2d1598094754dd8435c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2749326e8eca603a0b623e39b103548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_275856d243390f990a99c2c14d2e951c
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 50, 50], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()