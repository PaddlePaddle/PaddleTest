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
    class PrimitiveOp_91cd819b6bbaea0afd59057b9f8d78c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 24, 36]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c744737398ac6da94b86b3f5c5bf81bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91cd819b6bbaea0afd59057b9f8d78c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c744737398ac6da94b86b3f5c5bf81bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91cd819b6bbaea0afd59057b9f8d78c5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb81aeb8b8da86ab2350da1b0e74f92a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, -1, -1]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 150, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e426ff1ba9f4a6ddd70c9da789919f1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb81aeb8b8da86ab2350da1b0e74f92a
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_247420cc23812456375d9497911f3757(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 256, 21]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_142863af5b4883283fdc739790057485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_247420cc23812456375d9497911f3757
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_334c1bc64ea92680e2a3da5276091a61(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 25, 38]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2bc895b7ea7488acff3ed827fc04ca2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_334c1bc64ea92680e2a3da5276091a61
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bc895b7ea7488acff3ed827fc04ca2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_334c1bc64ea92680e2a3da5276091a61
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6fba2b6e9f74f8887f7c71c7603953b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 20, 30]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77f45b7a6bbc44981ea46179a126181c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fba2b6e9f74f8887f7c71c7603953b0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77f45b7a6bbc44981ea46179a126181c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fba2b6e9f74f8887f7c71c7603953b0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_407ea3c4910ea4eeb9ed1852b7988be6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, -1, -1]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a770fa71c2c6cd75795295e72aaa168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_407ea3c4910ea4eeb9ed1852b7988be6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b1e4474936391cc130998d907f3af92e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [7, 256, 19]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c462c0786e38807df770fad0329c1249(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1e4474936391cc130998d907f3af92e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a723359409b73b128687ce2fc793350e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 15, 25]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ec3136c03f02a99dbb62364a4b2f2e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723359409b73b128687ce2fc793350e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ec3136c03f02a99dbb62364a4b2f2e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723359409b73b128687ce2fc793350e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac6575c5c5b411e70ad2c1f8dccdf513(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, -1, -1]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 150, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff565b356805283555020658903e0bba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac6575c5c5b411e70ad2c1f8dccdf513
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_76d933b89cb1b958aa99f5dc66b5892c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [8, 256, 150]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d55d0bb6414caa9c284f15d7bbba5e84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76d933b89cb1b958aa99f5dc66b5892c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 150], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b11cc29e997953802640b67bbf9099e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 24, 36]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d48f4f41d9bd232c63c6f2e9d2cddda6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b11cc29e997953802640b67bbf9099e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d48f4f41d9bd232c63c6f2e9d2cddda6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b11cc29e997953802640b67bbf9099e
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba062171aead03c9736cfc31a7bafdca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, -1, -1]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09e4ba15ef38af46f1db088f26c5fa5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba062171aead03c9736cfc31a7bafdca
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fd3c44fc0c13f711bf727a7c154e6b98(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 256, 21]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74021e6bc32eea55f98de446919ab0ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd3c44fc0c13f711bf727a7c154e6b98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0a6b70339e723dc1414b65e3c7ce88a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 25, 38]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9278ede491620a3e4efca2dd094863ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a6b70339e723dc1414b65e3c7ce88a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9278ede491620a3e4efca2dd094863ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a6b70339e723dc1414b65e3c7ce88a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6001d669a018e4905d8a497d76795776(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 20, 30]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57a0d6eeab0d947e40269f6ec2d35cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6001d669a018e4905d8a497d76795776
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a0d6eeab0d947e40269f6ec2d35cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6001d669a018e4905d8a497d76795776
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_649ddea74d1e2da243e9c53e8d53618e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba062171aead03c9736cfc31a7bafdca
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be44362217c62063ac8f9d4878f7fba9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [7, 256, 19]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a78433ffff3a86922bf4b6dce9ceb763(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be44362217c62063ac8f9d4878f7fba9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_080ef7babf296464841c115c5afb248b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 15, 25]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6acda789eba1ddbe707102f81d49aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_080ef7babf296464841c115c5afb248b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6acda789eba1ddbe707102f81d49aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_080ef7babf296464841c115c5afb248b
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2e4f3132f816f300897fc0230cffe9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba062171aead03c9736cfc31a7bafdca
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ace0226123c7d9967829de0109c0c9ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [8, 256, 150]
            return paddle.expand(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b67dfdfb865cfbfca14ee331c7ed3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ace0226123c7d9967829de0109c0c9ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 150], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()