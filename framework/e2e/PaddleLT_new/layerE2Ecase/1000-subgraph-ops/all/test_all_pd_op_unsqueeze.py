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
    class PrimitiveOp_258aa9e82104fdd2c80497ff1d972885(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [0, 1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b09a4ed168f8bbbfa584332e63455e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b09a4ed168f8bbbfa584332e63455e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e0b9e149d1f31931f8c71d4599579ab4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c55233bfb528afd8b5006e85d1dba6fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0b9e149d1f31931f8c71d4599579ab4
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d65c922f817106b5047aec37f4cd0e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4bf39a557839d70ee2ce6986ee8ad6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d65c922f817106b5047aec37f4cd0e4
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0221139b628eea354db93efdc1410b88(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3ae0b3f955b6174f684ac3d077c37dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0221139b628eea354db93efdc1410b88
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a6c4de5e18671d16d3d79917901d58a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e83f00a0d22ca04b616ae41eac9104a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6c4de5e18671d16d3d79917901d58a0
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_40f0a65abe87bf46ad070f4293605b26(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 336], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09e86af1f83d311a00891907311b901e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40f0a65abe87bf46ad070f4293605b26
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4e6d6926c58f596b11657f628df18648(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c847cf57ac259707e5d348a3f0adc841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e6d6926c58f596b11657f628df18648
        def get_inputs(self):
            return [
                paddle.uniform([4, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7f1b6abdd5ba21618c365c595a0afea6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6a42bbf6d0a8b549e9ba2a29bab065b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f1b6abdd5ba21618c365c595a0afea6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.49104568362236023, 0.35629308223724365, 0.3221745789051056, 0.290326863527298], dtype='float32').reshape([4]),
            ]


    
    class PrimitiveOp_a48dd12f1024f85ca0e2ef1ecb645205(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 9, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27f0268b3c7317b905b7b6256c388cb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a48dd12f1024f85ca0e2ef1ecb645205
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3241ff9ddaf77f97701a8d21b6f4c3db(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc3d7c238c17bff9a01cceb09ec66149(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3241ff9ddaf77f97701a8d21b6f4c3db
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4fb9d7f9d77b087c7740824a9602a18(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f67a5f3b6dc21967add4a3bc117243a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4fb9d7f9d77b087c7740824a9602a18
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            ]


    
    class PrimitiveOp_cb56cea869e2d43a94b20d1a7bea259a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27c6488dfb2f419a2e73c458a4290e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb56cea869e2d43a94b20d1a7bea259a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            ]


    
    class PrimitiveOp_d6abbce91b60fd3d717ae8354660f26d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea904e2ae72d29465386588b92b76755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6abbce91b60fd3d717ae8354660f26d
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bc7ba73d62f827b7c320304939d703ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_632348790f119bc4580db481afb06ca5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1e1883474cca87aea87211ef5de5a1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40f0a65abe87bf46ad070f4293605b26
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07830589640fe8cfc79f8776d3d3f046(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89bf3f5d58ef89f0579dd87a33e4f60a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07830589640fe8cfc79f8776d3d3f046
        def get_inputs(self):
            return [
                paddle.uniform([21, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f67a5f3b6dc21967add4a3bc117243a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4fb9d7f9d77b087c7740824a9602a18
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            ]


    class TestPrimitiveOp_27c6488dfb2f419a2e73c458a4290e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb56cea869e2d43a94b20d1a7bea259a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            ]


    class TestPrimitiveOp_1e1883474cca87aea87211ef5de5a1fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40f0a65abe87bf46ad070f4293605b26
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_263b1c2d8522a40b4b0588d100c67bbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d423684849e34940b9825bdaaa2ed1ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a757fb9f5cbccce9e26c339754f718f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d423684849e34940b9825bdaaa2ed1ec
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2746506929397583, 0.3692845404148102], [0.2529679834842682, 0.3538026511669159], [0.1918400079011917, 0.46419987082481384], [0.4446069300174713, 0.48151710629463196], [0.22252815961837769, 0.22237402200698853], [0.07176125049591064, 0.38622698187828064]]], dtype='float32').reshape([1, 6, 2]),
            ]


    
    class PrimitiveOp_42b80852403b5da22f679bbdbb0d7bd9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 49, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_372614d483cf3cd6b0e8f68cf37ad252(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42b80852403b5da22f679bbdbb0d7bd9
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_65b0beba57ca4d11525e10ba61122a67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d52c3b47b6b29bb08d011d455c1eae59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_65b573158e2333372bee230e2d6d6771(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            ]


    
    class PrimitiveOp_c96429f74e9c48bbb921cdb0e28ba69e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bfc371e56c8e17545e97ae352221bc69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c96429f74e9c48bbb921cdb0e28ba69e
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2e40aa0561db9519a797617801226547(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee1c411a07fa21c3e52607eb4cb444a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e40aa0561db9519a797617801226547
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c795e50418028bf13345f2e08d21e7e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33cdd686b8a515509cb598e45e15294e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c795e50418028bf13345f2e08d21e7e2
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_72daf04aa7f37406644f972121b45b98(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2d8e539b34986ae366c411f15ca0d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72daf04aa7f37406644f972121b45b98
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.044002752751111984, 0.4828374981880188, 0.2906877398490906, 0.013120715506374836]], dtype='float32').reshape([1, 4]),
            ]


    
    class PrimitiveOp_a19b5aaf2b4575ab7d24a75975eb2bfd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3191f9796f87df02e93d484f308baa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a19b5aaf2b4575ab7d24a75975eb2bfd
        def get_inputs(self):
            return [
                paddle.uniform([300, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_16868655f0743686b9ee9a1e5551412c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_654e3ce92ca6b843476bc7ac86c25d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_654e3ce92ca6b843476bc7ac86c25d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04c366301655514ff6b55c2ddfd5e232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9e4df829da228bca983f809e00182303(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9eafd8388f4ea2a3b22ec5a3bb069108(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e4df829da228bca983f809e00182303
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05472e234d38316f7ea91393faed3070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05472e234d38316f7ea91393faed3070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3d2085997e754745e696ed60ddfd9f2e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 21], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e355a6d88bd7dcdbd46b2870e62b7289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d2085997e754745e696ed60ddfd9f2e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af7d81130de7497a150d178f9b4c5266(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e6d6926c58f596b11657f628df18648
        def get_inputs(self):
            return [
                paddle.uniform([3, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc98c4a832d6617b4c0515e395328775(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f1b6abdd5ba21618c365c595a0afea6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.21855682134628296, 0.10699958354234695, 0.44307953119277954], dtype='float32').reshape([3]),
            ]


    class TestPrimitiveOp_a941f20d073434347efd5c506ad00c54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6abbce91b60fd3d717ae8354660f26d
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_78190bba9f0fac895270ca092810b8b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 872], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_00f0b3aa1755187924a80e84d95e383f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78190bba9f0fac895270ca092810b8b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bb79a0043926ba6dac999526a19d7f71(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f45e4aa902ee2f102f9d1c8519704e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    
    class PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e78be695d432ac080b9a248a42dca717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([1827], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d4dfc8583c38a42e83df73098c1a862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_80314d07cfb324f5318bf89bc4072c62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe0dccd47ed596da22d12a70ed030642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_fe0dccd47ed596da22d12a70ed030642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_8e72c1b0dacfdc5c0bf4a5401a876b07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e72c1b0dacfdc5c0bf4a5401a876b07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_329a84f8090d99280ed512d679350423(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 4, 49, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ec3d44c4909adc98498a4242a37430d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_329a84f8090d99280ed512d679350423
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e34361c3dd5a0a402575ccbe5e418282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40f0a65abe87bf46ad070f4293605b26
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_54066934d4f8d99c9bc0112f3c21b5a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d2508d83064f07bffa11b1f1d57a02d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54066934d4f8d99c9bc0112f3c21b5a7
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab7735e00d3b765108093d95c4c821c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c96429f74e9c48bbb921cdb0e28ba69e
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38fcfed5a5530df87f198f881407b136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109], dtype='int32'),
            ]


    class TestPrimitiveOp_11316d53afa15ae60ff6c8e5ce52156a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([5514], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_288bbc530ae3b8894d0d3322367ac517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a1a264d4f15051113b32c2c38f453964(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_a1a264d4f15051113b32c2c38f453964(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_505812cd926760c5c2c4a8f95f87760d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_505812cd926760c5c2c4a8f95f87760d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_09e86af1f83d311a00891907311b901e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40f0a65abe87bf46ad070f4293605b26
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dfbc2d75c206056cd83d91555b25408f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 49, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87b3cf0a3ab6cdfe104558bd84ac823a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfbc2d75c206056cd83d91555b25408f
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_353262a1d1bd406516bdf5ede8b8061f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07830589640fe8cfc79f8776d3d3f046
        def get_inputs(self):
            return [
                paddle.uniform([19, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_41d0dd6ac88565905fe4c4ca532c6d68(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_173da2b31e47c584370f14db482a9e41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41d0dd6ac88565905fe4c4ca532c6d68
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_537bf00197ce46b26fe9671298844f1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6c4de5e18671d16d3d79917901d58a0
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45e4aa902ee2f102f9d1c8519704e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_9ab6db01714c4d8087c611e5e39ea9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([1799], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d4dfc8583c38a42e83df73098c1a862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_98376f43c1a74dc82912538917ef150b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_98376f43c1a74dc82912538917ef150b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_bf4e1ec02463435b186b5a5b76904219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40f0a65abe87bf46ad070f4293605b26
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c902e63c2dcb575c55687c427667d61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c96429f74e9c48bbb921cdb0e28ba69e
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e34361c3dd5a0a402575ccbe5e418282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40f0a65abe87bf46ad070f4293605b26
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a0986245dd678df2db2e9a808bbbb650(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dca499411df5fa3f8ed8f11e5c9da6b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0986245dd678df2db2e9a808bbbb650
        def get_inputs(self):
            return [
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3f253a897b215782057bdab0b864836(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_959a646ddc1e78590c3e3cf8676bbc31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3f253a897b215782057bdab0b864836
        def get_inputs(self):
            return [
                paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d5ee270371ac5b487c293ee0364bc278(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be12069885baf30ab6abfbc1dae7d4ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5ee270371ac5b487c293ee0364bc278
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c54531ec933d6e0d837e746f3141517(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 8, 49, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09051089cabf12da92ed43249e59b458(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c54531ec933d6e0d837e746f3141517
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06d0d249d5769b1cd31f24c49ef466cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_1c8c23ddda158fc1aa91784b94e4b3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_ba23fbee7523587cd4c325790108adec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6abbce91b60fd3d717ae8354660f26d
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a17849a354af71d7e9f33b96e3634b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024], dtype='int32'),
            ]


    class TestPrimitiveOp_31fdbd662ac1aca3cc79ad449bd120ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([1503], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62d4f4f9a30ec7b07ee4ce6873a18327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ab04bae9a76a6b5c39f3260c128d5c77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ab04bae9a76a6b5c39f3260c128d5c77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_d1f7c1cc2acaebe352bc3b26443bf9cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3241ff9ddaf77f97701a8d21b6f4c3db
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_029e4b271ecc2f5987365e9a647068c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 16, 49, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60770ddd9bb2b75c34382ff31227c3ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_029e4b271ecc2f5987365e9a647068c3
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be77f09a52aee891a1241e4e6fb229a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c96429f74e9c48bbb921cdb0e28ba69e
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93277a7b175ba1e7736ce92419301785(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_9b3bd450c1f9fd67881c25c61c803125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_04c366301655514ff6b55c2ddfd5e232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9eafd8388f4ea2a3b22ec5a3bb069108(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e4df829da228bca983f809e00182303
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc7cb5ce692f51e6c7fbb5fa0732425c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 97, 97], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ada578e655b31db78768e1017d63bfee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc7cb5ce692f51e6c7fbb5fa0732425c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb17dc2f00c815b821e698c6e468abbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41d0dd6ac88565905fe4c4ca532c6d68
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60770ddd9bb2b75c34382ff31227c3ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_029e4b271ecc2f5987365e9a647068c3
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04c366301655514ff6b55c2ddfd5e232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9eafd8388f4ea2a3b22ec5a3bb069108(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e4df829da228bca983f809e00182303
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffd2b6ffc7e475d48d4a6836fd192def(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6abbce91b60fd3d717ae8354660f26d
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd9558c4f9a8ec863711ec10f26c29de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd9558c4f9a8ec863711ec10f26c29de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a53a2342e02b1608d5bc31dc391142e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_494b386d8ab40911c44284dd37d5bf3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a53a2342e02b1608d5bc31dc391142e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5177f368b4b2653d772d0aafe2d7678f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3241ff9ddaf77f97701a8d21b6f4c3db
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf4e1ec02463435b186b5a5b76904219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40f0a65abe87bf46ad070f4293605b26
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09051089cabf12da92ed43249e59b458(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c54531ec933d6e0d837e746f3141517
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53e667d53d493c0702b22c2cd077f1f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_beb864cb9721f7545aa6375fdd480783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([2077], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bda84e1ce54ee4a0337926d42cf9fc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1f4dc8d181669d73700090e3ff2c5c19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_1f4dc8d181669d73700090e3ff2c5c19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_359dc6b849d13c9844e900eb3b0f6de3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261], dtype='int32'),
            ]


    class TestPrimitiveOp_9e0139dd5bd8179c9b74c7779ae30fb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([4628], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9e200b60b68b50340f5d08d2f14c395(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ece326b354bfa13d6cb2788a09f0c55c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_ece326b354bfa13d6cb2788a09f0c55c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_2138ef15f30ded231ddea8c10c1f11e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e6d6926c58f596b11657f628df18648
        def get_inputs(self):
            return [
                paddle.uniform([6, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cda9926d9ae065417a8e153e17e9e991(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f1b6abdd5ba21618c365c595a0afea6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4964856207370758, 0.40626269578933716, 0.46424606442451477, 0.3560028672218323, 0.10537204146385193, 0.3426622450351715], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ef53ae2832b64897cac9293062246d16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_510511ada31860901299dbb250dc81d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_f770563d462dc7db0bc37bbada7afdf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([1101], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3ba7e8f4de869ae5aae6766093b45a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c52d8c9e029dd5c238b6005b36698a6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_c52d8c9e029dd5c238b6005b36698a6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_e094e3abfb72be48602c2833aa46d78e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [0]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77d92be4a29017c34284c862324a9608(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e094e3abfb72be48602c2833aa46d78e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37339529395103455, 0.30681005120277405, 0.14150168001651764, 0.0019667954184114933], dtype='float32').reshape([4]),
            ]


    
    class PrimitiveOp_c6d19c76b8bea8827820e854b4cca430(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14940e9812706774904a64739aeb346f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d19c76b8bea8827820e854b4cca430
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.37339529395103455, 0.30681005120277405, 0.14150168001651764, 0.0019667954184114933]], dtype='float32').reshape([1, 4]),
            ]


    
    class PrimitiveOp_82eb4b5c34660af478cb26be4bd0efd1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_124a68a4823ae58a221be7b0efdd5fc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82eb4b5c34660af478cb26be4bd0efd1
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b324e264f55fe8adcbfd4813d677d22d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54066934d4f8d99c9bc0112f3c21b5a7
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_21a6cd7bcd889d2a3ff199c2ddf0b32a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 49, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1de0dc058ce649d9b5715d0e0542e8c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21a6cd7bcd889d2a3ff199c2ddf0b32a
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04c366301655514ff6b55c2ddfd5e232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9eafd8388f4ea2a3b22ec5a3bb069108(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e4df829da228bca983f809e00182303
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ecc351c71d5a7ab91533674b74aae9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e094e3abfb72be48602c2833aa46d78e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.015704719349741936, 0.1316700279712677, 0.0756133571267128, 0.4947950839996338], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_3906f28143984a16dddd67bec73b8721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d19c76b8bea8827820e854b4cca430
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015704719349741936, 0.1316700279712677, 0.0756133571267128, 0.4947950839996338]], dtype='float32').reshape([1, 4]),
            ]


    
    class PrimitiveOp_a12b00927fd6406049c1586675a0117c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_66030483e6004bc846d8850f92d9b716(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a12b00927fd6406049c1586675a0117c
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3d405e4efec1e4cecd32380009976979(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1248], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64755fbf3fca074c2ad3dfaa56cc8b45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d405e4efec1e4cecd32380009976979
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e53af632f2380093a7191042ce172ddc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6c4de5e18671d16d3d79917901d58a0
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3dd3f22d6c85b73f928bc11edf7e6a26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41d0dd6ac88565905fe4c4ca532c6d68
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ce55718e02b388cdec3c82472f2e061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ce55718e02b388cdec3c82472f2e061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f57ae66b3137f2a8084555aa2190878a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f57ae66b3137f2a8084555aa2190878a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb57af29b1a5fbf642dc3dee7b16e764(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725], dtype='int32'),
            ]


    class TestPrimitiveOp_d95f3656461e190220afb47c00403f57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([2361], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9be3dc94f897e66914604c0034369ddd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e910a92cdcfdf00d7ff1346df77b95ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_e910a92cdcfdf00d7ff1346df77b95ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_172084183a997eeb4336ecec94d5ca92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_172084183a997eeb4336ecec94d5ca92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_453b0a42b4b4ce3768451daf67605e8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069], dtype='int32'),
            ]


    class TestPrimitiveOp_6292558dbed03236f7dac90bf60cd824(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([3061], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ba6472723fcd3376a916ec45fd1cd1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_be6e532afb1b9fd4f5ef8a937323ed44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_be6e532afb1b9fd4f5ef8a937323ed44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_0deed171b0f9bc1878667f367031c9e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581], dtype='int32'),
            ]


    class TestPrimitiveOp_e64d864e7b8cc4783034791670127899(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([3799], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_644cf0c6b64fc62ad2d576f8c2026838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_428d382455f72877dc6584719de17388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_428d382455f72877dc6584719de17388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_6e49624f488fc733e0e435462f6dabb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e49624f488fc733e0e435462f6dabb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4df49f51e23bc59de59974dcc16f4d56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09c2d71e9dac00b68d5cc9c9230cbcdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4df49f51e23bc59de59974dcc16f4d56
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c33e17ad8ba72034a453acc7a356bf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c33e17ad8ba72034a453acc7a356bf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00f0b3aa1755187924a80e84d95e383f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78190bba9f0fac895270ca092810b8b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1645db0944ad18aaa0506b647e2c9652(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 49, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4f735135b8536ee677676eaf5af5667(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1645db0944ad18aaa0506b647e2c9652
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_336d3c429c3dfc91678100e2dbdd1603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6c4de5e18671d16d3d79917901d58a0
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf3caeccd04134de1d7792799872f837(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6c4de5e18671d16d3d79917901d58a0
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2ea6077afc5dc854eda6b18f3ec977b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e6d6926c58f596b11657f628df18648
        def get_inputs(self):
            return [
                paddle.uniform([2, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e250bd7d95f7c642c137a72cf0a95e69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f1b6abdd5ba21618c365c595a0afea6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4264984428882599, 0.44378095865249634], dtype='float32').reshape([2]),
            ]


    class TestPrimitiveOp_4d621a9184f2c59b518fca8d4839deae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41d0dd6ac88565905fe4c4ca532c6d68
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_055bec4c203d61560eaf54744c03006e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 49, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a635d30f0d72f9259c8a34ed1e83f73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_055bec4c203d61560eaf54744c03006e
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_54eecbfc4dee1aad78dafdb5debd5ff4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b835422df21a459416372a82432ef412(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54eecbfc4dee1aad78dafdb5debd5ff4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87b3cf0a3ab6cdfe104558bd84ac823a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dfbc2d75c206056cd83d91555b25408f
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9be511c85eefa4a132c66813511be961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_c359835a40a10346a0f542326982c6e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_494b386d8ab40911c44284dd37d5bf3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a53a2342e02b1608d5bc31dc391142e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4f735135b8536ee677676eaf5af5667(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1645db0944ad18aaa0506b647e2c9652
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53e667d53d493c0702b22c2cd077f1f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_8998be4bac39ef8a1ebc9d029f9874ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([2088], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bda84e1ce54ee4a0337926d42cf9fc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_2c08265091d65b75feaa743220546b9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_2c08265091d65b75feaa743220546b9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_8f67a5f3b6dc21967add4a3bc117243a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4fb9d7f9d77b087c7740824a9602a18
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            ]


    class TestPrimitiveOp_27c6488dfb2f419a2e73c458a4290e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb56cea869e2d43a94b20d1a7bea259a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            ]


    
    class PrimitiveOp_9f5bb742395fb31c37f63ee5752d4858(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2, 9, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c819881d95eba451183cb045288e64a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f5bb742395fb31c37f63ee5752d4858
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1de0dc058ce649d9b5715d0e0542e8c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21a6cd7bcd889d2a3ff199c2ddf0b32a
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dabc1429db3b0248b927ae3ba7dcd027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07830589640fe8cfc79f8776d3d3f046
        def get_inputs(self):
            return [
                paddle.uniform([150, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb47cd73576f67f5002a7f181aea3f62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400], dtype='int32'),
            ]


    class TestPrimitiveOp_3b8a1beee22ff2e450640de14bea70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([4270], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c65600bd73236a64d550d07dd2105134(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5287997bdc576f29411d9a602ac5af59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_5287997bdc576f29411d9a602ac5af59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80314d07cfb324f5318bf89bc4072c62
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_3dcaf14182663c024eb64c15eae82bc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 624], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a96b90c6ad6d5c0e1894fbdfea1a8dbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dcaf14182663c024eb64c15eae82bc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a635d30f0d72f9259c8a34ed1e83f73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_055bec4c203d61560eaf54744c03006e
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b09a4ed168f8bbbfa584332e63455e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b09a4ed168f8bbbfa584332e63455e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_202f8337edf9469ab63cf6995717d042(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0128682d962c10104eacf64e9d68d960(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ed9b03b172e7de20d727505c85c6e98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e34a74409fa5ce78929f27920583428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88e9bc4644f1d01bf640542a620a2ff5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0ff17c2f66b0889b6a2df3af0dd135a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dbef48a3a0db7b1e016c36078502129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([4, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a42bbf6d0a8b549e9ba2a29bab065b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f1b6abdd5ba21618c365c595a0afea6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.49104568362236023, 0.35629308223724365, 0.3221745789051056, 0.290326863527298], dtype='float32').reshape([4]),
            ]


    
    class PrimitiveOp_4f8494f75dc4c965561003666399f1f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c07898944b555d0b68f083fff98038dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc3d7c238c17bff9a01cceb09ec66149(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3241ff9ddaf77f97701a8d21b6f4c3db
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f67a5f3b6dc21967add4a3bc117243a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4fb9d7f9d77b087c7740824a9602a18
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            ]


    class TestPrimitiveOp_27c6488dfb2f419a2e73c458a4290e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb56cea869e2d43a94b20d1a7bea259a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            ]


    class TestPrimitiveOp_3f839897f9d8272b5bb2d9d07e2818e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_632348790f119bc4580db481afb06ca5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0de3439140c0e28c6772dd1258623a9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89bf3f5d58ef89f0579dd87a33e4f60a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07830589640fe8cfc79f8776d3d3f046
        def get_inputs(self):
            return [
                paddle.uniform([21, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f67a5f3b6dc21967add4a3bc117243a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4fb9d7f9d77b087c7740824a9602a18
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            ]


    class TestPrimitiveOp_27c6488dfb2f419a2e73c458a4290e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb56cea869e2d43a94b20d1a7bea259a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            ]


    class TestPrimitiveOp_0de3439140c0e28c6772dd1258623a9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_263b1c2d8522a40b4b0588d100c67bbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a757fb9f5cbccce9e26c339754f718f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d423684849e34940b9825bdaaa2ed1ec
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2746506929397583, 0.3692845404148102], [0.2529679834842682, 0.3538026511669159], [0.1918400079011917, 0.46419987082481384], [0.4446069300174713, 0.48151710629463196], [0.22252815961837769, 0.22237402200698853], [0.07176125049591064, 0.38622698187828064]]], dtype='float32').reshape([1, 6, 2]),
            ]


    class TestPrimitiveOp_3c0ec49dca8267a689761c7fd290bf17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d52c3b47b6b29bb08d011d455c1eae59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_65b573158e2333372bee230e2d6d6771(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_b955785b4c2a69dc2b57c968b3195b97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d01e0eed5e694eecff3f8d5a6be67fe5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce4d653c924dc86afe2eeff71eabd400(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d01e0eed5e694eecff3f8d5a6be67fe5
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcd76d5948add6e47649b839b3b79ea8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07830589640fe8cfc79f8776d3d3f046
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2d8e539b34986ae366c411f15ca0d42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72daf04aa7f37406644f972121b45b98
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.044002752751111984, 0.4828374981880188, 0.2906877398490906, 0.013120715506374836]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_62315eb0223d39e3df25e4f70e17e1ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07830589640fe8cfc79f8776d3d3f046
        def get_inputs(self):
            return [
                paddle.uniform([300, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_654e3ce92ca6b843476bc7ac86c25d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_654e3ce92ca6b843476bc7ac86c25d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04c366301655514ff6b55c2ddfd5e232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a039bd4b8c4c8da1aa295cce8505b209(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_721b53f54ce16125afb1aac04aae0ed0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a039bd4b8c4c8da1aa295cce8505b209
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05472e234d38316f7ea91393faed3070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05472e234d38316f7ea91393faed3070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8a4c2004c2b36928747c6248b8216b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d01e0eed5e694eecff3f8d5a6be67fe5
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c14bf37b75f390ad31c93f9ba0db7455(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([3, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc98c4a832d6617b4c0515e395328775(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f1b6abdd5ba21618c365c595a0afea6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.21855682134628296, 0.10699958354234695, 0.44307953119277954], dtype='float32').reshape([3]),
            ]


    class TestPrimitiveOp_0e17af61db1c5a9eb38b6c695f42f91a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bede2135c39894b73930424c82397727(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45e4aa902ee2f102f9d1c8519704e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_e78be695d432ac080b9a248a42dca717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([1827], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d4dfc8583c38a42e83df73098c1a862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_c950852d175717ed26ec41f23d63b091(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98b1781048019f658373590de4fb9862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_98b1781048019f658373590de4fb9862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_8e72c1b0dacfdc5c0bf4a5401a876b07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e72c1b0dacfdc5c0bf4a5401a876b07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9444614a8e96b34d5f4cdd6faa23bd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_997c43bd965d42c59841b26b593da6f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_262bb4cb0d7f2cc48ffd1c5731d67cc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d423684849e34940b9825bdaaa2ed1ec
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55621192866264e57128e2ef315270c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38fcfed5a5530df87f198f881407b136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109], dtype='int32'),
            ]


    class TestPrimitiveOp_11316d53afa15ae60ff6c8e5ce52156a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([5514], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_288bbc530ae3b8894d0d3322367ac517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_bd4f342386c1d8d319f6e126ba9df5b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_bd4f342386c1d8d319f6e126ba9df5b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_505812cd926760c5c2c4a8f95f87760d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_505812cd926760c5c2c4a8f95f87760d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_b0ff17c2f66b0889b6a2df3af0dd135a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0750ba84b7f32d6d688a1c3a7c3c4b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_353262a1d1bd406516bdf5ede8b8061f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07830589640fe8cfc79f8776d3d3f046
        def get_inputs(self):
            return [
                paddle.uniform([19, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aac0690cabf6025bcd518a89c95e4465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae5955fd55e5fde7f716892d0d78b80f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45e4aa902ee2f102f9d1c8519704e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_9ab6db01714c4d8087c611e5e39ea9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([1799], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d4dfc8583c38a42e83df73098c1a862(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_38d30b182d121b05b439eda173417600(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_38d30b182d121b05b439eda173417600(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_379e60c5add968d7e2d572f5e6afc058(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6764ab34b453d82537c289fda2714310(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_997c43bd965d42c59841b26b593da6f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e427fcc848fa5c56f0214c1f49fda08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e094e3abfb72be48602c2833aa46d78e
        def get_inputs(self):
            return [
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f961acd84ed8f0a2b253ca6730eaceab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3aa7bc9ce61296f374c467f11441e2ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f961acd84ed8f0a2b253ca6730eaceab
        def get_inputs(self):
            return [
                paddle.uniform([1, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1355d4528589fa5d17a3f43f12e9f3de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9610a9ee13ffb357cbd1d33b2348a805(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1355d4528589fa5d17a3f43f12e9f3de
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dec17c308819ca6c68e26d50b2cbcf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06d0d249d5769b1cd31f24c49ef466cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_1c8c23ddda158fc1aa91784b94e4b3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_bfe927797e9e6b99abb1c14a61adb2ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a17849a354af71d7e9f33b96e3634b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024], dtype='int32'),
            ]


    class TestPrimitiveOp_31fdbd662ac1aca3cc79ad449bd120ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([1503], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62d4f4f9a30ec7b07ee4ce6873a18327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5207f3e0a8cbb6e84e61ba4f86e2bee3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_5207f3e0a8cbb6e84e61ba4f86e2bee3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_d1f7c1cc2acaebe352bc3b26443bf9cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3241ff9ddaf77f97701a8d21b6f4c3db
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1058348e8f6a49b33448e459741c4091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3f4400eae8b79ccf4a3e6ccfc28af6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93277a7b175ba1e7736ce92419301785(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_9b3bd450c1f9fd67881c25c61c803125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_04c366301655514ff6b55c2ddfd5e232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721b53f54ce16125afb1aac04aae0ed0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a039bd4b8c4c8da1aa295cce8505b209
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d574a840657860b007cd28ca3d0745c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a039bd4b8c4c8da1aa295cce8505b209
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9963a0ae5b7d81b8df199da4af755a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1058348e8f6a49b33448e459741c4091(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04c366301655514ff6b55c2ddfd5e232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721b53f54ce16125afb1aac04aae0ed0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a039bd4b8c4c8da1aa295cce8505b209
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81cacc731366f557d6b17431fc146798(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd9558c4f9a8ec863711ec10f26c29de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd9558c4f9a8ec863711ec10f26c29de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_809ddaa8da5732caad9e4ba4cd3f15eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5177f368b4b2653d772d0aafe2d7678f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3241ff9ddaf77f97701a8d21b6f4c3db
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_379e60c5add968d7e2d572f5e6afc058(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dec17c308819ca6c68e26d50b2cbcf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53e667d53d493c0702b22c2cd077f1f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_beb864cb9721f7545aa6375fdd480783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([2077], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bda84e1ce54ee4a0337926d42cf9fc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c12ebb21f463bc111f5f4e188e849351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_c12ebb21f463bc111f5f4e188e849351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_359dc6b849d13c9844e900eb3b0f6de3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261], dtype='int32'),
            ]


    class TestPrimitiveOp_9e0139dd5bd8179c9b74c7779ae30fb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([4628], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9e200b60b68b50340f5d08d2f14c395(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9b0ed32d591a0bcd31875cdf0531d71c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_9b0ed32d591a0bcd31875cdf0531d71c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_c4a2dbb1be53a7385dbf749242efb016(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([6, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cda9926d9ae065417a8e153e17e9e991(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f1b6abdd5ba21618c365c595a0afea6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4964856207370758, 0.40626269578933716, 0.46424606442451477, 0.3560028672218323, 0.10537204146385193, 0.3426622450351715], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_ef53ae2832b64897cac9293062246d16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_510511ada31860901299dbb250dc81d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_f770563d462dc7db0bc37bbada7afdf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([1101], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3ba7e8f4de869ae5aae6766093b45a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_896dffdc87ffa2bf41261f0928b63a1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_896dffdc87ffa2bf41261f0928b63a1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_77d92be4a29017c34284c862324a9608(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e094e3abfb72be48602c2833aa46d78e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37339529395103455, 0.30681005120277405, 0.14150168001651764, 0.0019667954184114933], dtype='float32').reshape([4]),
            ]


    
    class PrimitiveOp_64971c274cab564232fc7adb12a75153(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1]
            return paddle.unsqueeze(input_0, input_1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0335f6e57ecf4b212b4d7b6503a2a0de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64971c274cab564232fc7adb12a75153
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.37339529395103455, 0.30681005120277405, 0.14150168001651764, 0.0019667954184114933]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_9926e21b81b8941698976f8cd6401b28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72daf04aa7f37406644f972121b45b98
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4da52d930dd8034f64b701e96cd9cd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d423684849e34940b9825bdaaa2ed1ec
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bece9053cce570a3b843dfd70a7b2ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04c366301655514ff6b55c2ddfd5e232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d08ff010f48a563c5a32cc6b595ce19b
        def get_inputs(self):
            return [
                paddle.uniform([512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721b53f54ce16125afb1aac04aae0ed0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a039bd4b8c4c8da1aa295cce8505b209
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ecc351c71d5a7ab91533674b74aae9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e094e3abfb72be48602c2833aa46d78e
        def get_inputs(self):
            return [
                paddle.to_tensor([0.015704719349741936, 0.1316700279712677, 0.0756133571267128, 0.4947950839996338], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_f47c29b15f85cf9370de50fca89667c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64971c274cab564232fc7adb12a75153
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.015704719349741936, 0.1316700279712677, 0.0756133571267128, 0.4947950839996338]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_1af4eef68f7b48d75052d93c993c8226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72daf04aa7f37406644f972121b45b98
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a493a97aa6a7fa9b5bf335ef6d88f4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92d2fc5d534db1f6d47965358956ab98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d5992aafdae1aef5463e9ab33d1c8cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ce55718e02b388cdec3c82472f2e061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ce55718e02b388cdec3c82472f2e061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258aa9e82104fdd2c80497ff1d972885
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f57ae66b3137f2a8084555aa2190878a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f57ae66b3137f2a8084555aa2190878a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb57af29b1a5fbf642dc3dee7b16e764(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725], dtype='int32'),
            ]


    class TestPrimitiveOp_d95f3656461e190220afb47c00403f57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([2361], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9be3dc94f897e66914604c0034369ddd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_acbf5b75fc6eefb2afa121e0cb4497e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_acbf5b75fc6eefb2afa121e0cb4497e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_172084183a997eeb4336ecec94d5ca92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_172084183a997eeb4336ecec94d5ca92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_453b0a42b4b4ce3768451daf67605e8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069], dtype='int32'),
            ]


    class TestPrimitiveOp_6292558dbed03236f7dac90bf60cd824(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([3061], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ba6472723fcd3376a916ec45fd1cd1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_2cea74ea0f6ef77f5bf3a76d2e059b1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_2cea74ea0f6ef77f5bf3a76d2e059b1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_0deed171b0f9bc1878667f367031c9e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581], dtype='int32'),
            ]


    class TestPrimitiveOp_e64d864e7b8cc4783034791670127899(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([3799], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_644cf0c6b64fc62ad2d576f8c2026838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c8ebe5333b50d20a9e91cc05558f4b33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_c8ebe5333b50d20a9e91cc05558f4b33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_6e49624f488fc733e0e435462f6dabb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e49624f488fc733e0e435462f6dabb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cbd7b10967b939ef2c0f3ceea0bf660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c33e17ad8ba72034a453acc7a356bf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c33e17ad8ba72034a453acc7a356bf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16868655f0743686b9ee9a1e5551412c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bede2135c39894b73930424c82397727(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6d857888daedb782a456d29acdcf2c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a7c56f7c604989a5875fbcd397a7696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa7a6c2ec735e92cced153c00dab457e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_114086aa385b565d9bca3dd330ed934e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([2, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e250bd7d95f7c642c137a72cf0a95e69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f1b6abdd5ba21618c365c595a0afea6
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4264984428882599, 0.44378095865249634], dtype='float32').reshape([2]),
            ]


    class TestPrimitiveOp_a8117f5750195f82b923cebd780cfd56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82d312360752d50b1558e8fb5d59034a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47c2a4ec9c2b906ab8d2a5f4a98f7f4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0750ba84b7f32d6d688a1c3a7c3c4b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9be511c85eefa4a132c66813511be961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_c359835a40a10346a0f542326982c6e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65b0beba57ca4d11525e10ba61122a67
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_809ddaa8da5732caad9e4ba4cd3f15eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6d857888daedb782a456d29acdcf2c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53e667d53d493c0702b22c2cd077f1f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_8998be4bac39ef8a1ebc9d029f9874ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([2088], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bda84e1ce54ee4a0337926d42cf9fc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1374e93c81666b2f11bf8ddadd6114d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_1374e93c81666b2f11bf8ddadd6114d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_8f67a5f3b6dc21967add4a3bc117243a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4fb9d7f9d77b087c7740824a9602a18
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int64'),
            ]


    class TestPrimitiveOp_27c6488dfb2f419a2e73c458a4290e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb56cea869e2d43a94b20d1a7bea259a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500], dtype='int32'),
            ]


    class TestPrimitiveOp_542f6fa909b6afbb34e89169acfdda0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bece9053cce570a3b843dfd70a7b2ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dabc1429db3b0248b927ae3ba7dcd027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07830589640fe8cfc79f8776d3d3f046
        def get_inputs(self):
            return [
                paddle.uniform([150, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb47cd73576f67f5002a7f181aea3f62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb79a0043926ba6dac999526a19d7f71
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400], dtype='int32'),
            ]


    class TestPrimitiveOp_3b8a1beee22ff2e450640de14bea70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_293d217fcacf1f64da2b9a4c12b5b5b4
        def get_inputs(self):
            return [
                paddle.uniform([4270], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c65600bd73236a64d550d07dd2105134(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc7ba73d62f827b7c320304939d703ae
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b672c662896783d4211c71d9188d48d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_b672c662896783d4211c71d9188d48d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c950852d175717ed26ec41f23d63b091
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_b73dd40465606d7973bcfd4d438855be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202f8337edf9469ab63cf6995717d042
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82d312360752d50b1558e8fb5d59034a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f8494f75dc4c965561003666399f1f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()