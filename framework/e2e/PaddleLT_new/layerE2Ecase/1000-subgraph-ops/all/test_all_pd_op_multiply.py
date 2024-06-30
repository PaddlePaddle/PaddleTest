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
    class PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd91351616bbf052fac48f1d09f0c823(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6b7732d26e82b6afe3f760a0e9a339c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6b7732d26e82b6afe3f760a0e9a339c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2a1cd420d046be81d9368d11da27a448(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 576, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4e637586a93d84db6b90280d72e4970(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a1cd420d046be81d9368d11da27a448
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_39eeb4c4229adf52e1c69a9d7b0416e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8103394a886de281dc1dacf0b64a5ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39eeb4c4229adf52e1c69a9d7b0416e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f1bca2cddd56f5355a3783af414d7fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 92, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7bece3e68c1ee2d13625cdd8418fcfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f1bca2cddd56f5355a3783af414d7fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa4b5c3f8a24ee89d4a52d0f2b5122ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0029604582a10483549d3f1c94229675(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 160, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aff1caeac10ff39894eb7b5aef77f841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0029604582a10483549d3f1c94229675
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3404b86090996e7cd122523633fc4f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5092c3ce222750f129b36460b4c325e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    
    class PrimitiveOp_bbd624b2f425754566721f11b1e8945b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 768, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ec3dcfd4a9d9f6d8c1005221ca6c245(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbd624b2f425754566721f11b1e8945b
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d81e2ce215128aabf309d06c1acf0f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25ccedbccb8ed857ef3ae38139254b52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d81e2ce215128aabf309d06c1acf0f1
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5c2beaf3386cd8866d12f617452b801(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f72ee08bd49926f5a12dc9ea61f4d8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b23176ab5027fa426d691ae4fbbd44b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f72ee08bd49926f5a12dc9ea61f4d8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5a83857ed4ec58facbf0dc551ac3723(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0cda5daef6722766839f36b0a6eaf4fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12418050318956375], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fed5d7cc32f3b8e54c41f1f79ff4218e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_06fe9d161f929580f6f7960ca98450bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_766f59d7272926465c7c97c870608529(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6675dd8393b353de2d380e5bafc030fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8656438049c1a0842dfa1a56c09a660(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b96de21277dd171f9c0885bb7c1696cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8656438049c1a0842dfa1a56c09a660
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_9fd6430838e0392695e74e050e2fc9f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7828f763e084720219866f26dc37b9e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fd6430838e0392695e74e050e2fc9f8
        def get_inputs(self):
            return [
                paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.49104568362236023]], [[0.35629308223724365]], [[0.3221745789051056]], [[0.290326863527298]]], dtype='float32').reshape([4, 1, 1]),
            ]


    
    class PrimitiveOp_7bce69f153b2ef7583e2f77ba3767511(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a1d86141670ee6757c5e04c2f48764f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7bce69f153b2ef7583e2f77ba3767511
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 72, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e1097015c5d61a3e31d3b2437223223(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_479de46fc86cacc3d09389795a955fa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ea05d03ea7172350a3509fe0168cabf0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 1, 9, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dea6eb472369a132e3490910a19195bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ea05d03ea7172350a3509fe0168cabf0
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7915ca315d9da77816b01607415ef2f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9557bb5c2e2153629125cae28053bb00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2802686095237732], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_58d05c01baf5469b79e4d75973043a16(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee0d9d192c2f846a44b901c078545359(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58d05c01baf5469b79e4d75973043a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.9013332724571228]], [[0.8568813800811768]], [[0.8735890984535217]], [[0.9094127416610718]], [[0.7574430108070374]], [[0.9139612913131714]], [[0.9421039819717407]], [[0.8919312357902527]], [[0.87510746717453]], [[0.8356452584266663]], [[0.9212054014205933]], [[0.8290369510650635]], [[0.8940802812576294]], [[0.7877908945083618]], [[0.935991644859314]], [[0.8388787508010864]], [[0.9542338848114014]], [[0.889322817325592]], [[0.8550490736961365]], [[0.8559415340423584]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_c0c4179364846592b0ea1dbad6fb15cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dfdd9d5d4b6878ced3b2f402a933dccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0c4179364846592b0ea1dbad6fb15cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 384, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c36e48a739a24e8d9c9fe39155d3e487(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7c7520a0add26c7ec0df49fd13cd683f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 960, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8566b6c5b896c4f7b725e7f01f954d59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c7520a0add26c7ec0df49fd13cd683f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_681cd2ef940523a3839b4fe0b203d716(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_080d7e236061cd000e50cc90ac6ee7d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_080d7e236061cd000e50cc90ac6ee7d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a5a4f5be8ada73ef594e3aeb38ea4d94(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c9476f5579ff036de8319386329cad0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5a4f5be8ada73ef594e3aeb38ea4d94
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24372179806232452]]], dtype='float32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_834b573ab8879e50b2da5dbf60a0f05e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2100, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_679d81ba42fc803d5a0a51816b916202(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_834b573ab8879e50b2da5dbf60a0f05e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b7a97039d8f2e516240a5b4c5e50323(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.06334684044122696], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_09d030a477e87212872e549e516580ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_461078821b823568be9e3706294fa19f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d37c59f56bf6b74141893902d22eb3aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c448bc0dd0aacad252e82f230b6360e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 60, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98a8ab972c6929c606a58bf0c67ca0db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c448bc0dd0aacad252e82f230b6360e3
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_33331224b71f35ab09d274602959b89e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb4a7587a0a3284f774fee668c0ccce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cf3343ba9b9329e68e46897656e4ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3033e60c17d4e36b32b9f1d241e4c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_caafe537c56accd0571e656c4016f429(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c7520a0add26c7ec0df49fd13cd683f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe916e29c7e10c5c102f97cad3eb9ff5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a1d990179688d9253d7d9521a5509f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1152, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1152, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2beee6623380a90ab9945b3b4a287d74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a1d990179688d9253d7d9521a5509f6
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df6038c9a94a07acd90053988d474464(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 192, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1d3d60e14b7fe7ece17ec32bcb740a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80e69fd54459208cdc3246f4300c3e98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2308b4986ef7dbb3cdff0f5e81eba37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 68, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4ad2396efba5ca68f18ae209be85a3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.221171572804451], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5aeee2de41b3ff23fe2dfbd8948b17de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b088ba409b7fc490bcff3aca7b1c1847(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f4eb25ed9d0aefe2827f915a77c1477(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8656438049c1a0842dfa1a56c09a660
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92f331e46ce1daacd30a006fe908aa82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_900fe0fe989902aa61116f9dbe128077(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4209981858730316], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6fc2057812ad851d956c4d9606f2ebde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_480c7d6d76b1578ad94bdef5a8b5a915(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 16, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e957210a228c2ca9c306aa2630d5c42b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480c7d6d76b1578ad94bdef5a8b5a915
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.6297255158424377]], [[0.9214640259742737]], [[0.8490052819252014]], [[0.7044241428375244]], [[0.825300931930542]], [[0.770827054977417]], [[0.6619652509689331]], [[0.7667979598045349]], [[0.7892367839813232]], [[0.9679443836212158]], [[0.819905161857605]], [[0.6637662053108215]], [[0.921315610408783]], [[0.720565676689148]], [[0.8479679226875305]], [[0.728390634059906]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_bbb8783744231779eb3416a311f96cbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18686c92b5c75e2ab6832de6b7f564b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8656438049c1a0842dfa1a56c09a660
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0bd285d471000535c0091c333342a48(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 44, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 44, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5a99632db6681a0253f0eb01195b880a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0bd285d471000535c0091c333342a48
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d5ba65aab3f930f78c1cb39bb2422d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0029604582a10483549d3f1c94229675
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71ac099fe8b14c90f0a00a455feaedac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.12735623121261597, 0.1039394810795784]], [[-0.010912656784057617, 0.05330643057823181]], [[-0.048233628273010254, 0.2193552404642105]], [[0.2782248854637146, -0.4064857065677643]], [[-0.1268930435180664, 0.32784441113471985]], [[0.19737398624420166, 0.012657210230827332]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_ecedf5ec28eb8cffd970c6f31df7563e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[-0.06997059285640717, 0.07948741316795349]], [[-0.27302616834640503, 0.08186653256416321]], [[0.12233045697212219, 0.23984484374523163]], [[-0.05789706110954285, -0.15295448899269104]], [[0.11353802680969238, 0.14615774154663086]], [[0.1804485023021698, 0.03878364711999893]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_e7917e9a00157f4d082a9e82af1a60c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.12735623121261597, 0.1039394810795784]], [[-0.010912656784057617, 0.05330643057823181]], [[-0.048233628273010254, 0.2193552404642105]], [[0.2782248854637146, -0.4064857065677643]], [[-0.1268930435180664, 0.32784441113471985]], [[0.19737398624420166, 0.012657210230827332]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.12735623121261597, 0.1039394810795784]], [[-0.010912656784057617, 0.05330643057823181]], [[-0.048233628273010254, 0.2193552404642105]], [[0.2782248854637146, -0.4064857065677643]], [[-0.1268930435180664, 0.32784441113471985]], [[0.19737398624420166, 0.012657210230827332]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_6f4f98b3264a451a39d41af1110cc320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[-0.06997059285640717, 0.07948741316795349]], [[-0.27302616834640503, 0.08186653256416321]], [[0.12233045697212219, 0.23984484374523163]], [[-0.05789706110954285, -0.15295448899269104]], [[0.11353802680969238, 0.14615774154663086]], [[0.1804485023021698, 0.03878364711999893]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[-0.06997059285640717, 0.07948741316795349]], [[-0.27302616834640503, 0.08186653256416321]], [[0.12233045697212219, 0.23984484374523163]], [[-0.05789706110954285, -0.15295448899269104]], [[0.11353802680969238, 0.14615774154663086]], [[0.1804485023021698, 0.03878364711999893]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_a2fa3e46a3e0a2407563783c01fdad9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5a4f5be8ada73ef594e3aeb38ea4d94
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.004442228935658932], [0.0001610954204807058], [0.011329323053359985], [0.1195206269621849], [0.04344525560736656], [0.007736477535218]]], dtype='float32').reshape([1, 6, 1]),
                paddle.to_tensor([[[0.02352176047861576], [0.1643000692129135], [0.1283266544342041], [0.1822909265756607], [0.12502682209014893], [0.02151423506438732]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_cb97f3214ec652f0f9b62b1a579b98d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5a4f5be8ada73ef594e3aeb38ea4d94
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.0011875410564243793], [0.02315785363316536], [0.01951729878783226], [0.004374376963824034], [0.006339387036859989], [0.0062875086441636086]]], dtype='float32').reshape([1, 6, 1]),
                paddle.to_tensor([[[0.02352176047861576], [0.1643000692129135], [0.1283266544342041], [0.1822909265756607], [0.12502682209014893], [0.02151423506438732]]], dtype='float32').reshape([1, 6, 1]),
            ]


    
    class PrimitiveOp_231d7fcaa6e7ec7976e960da177ad71e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6be595066291d7a4b24eb793f891b03d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231d7fcaa6e7ec7976e960da177ad71e
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f38193909830512f070f02c0019a55a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 1, 49, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7c8c2d99d847088cb1adc2e4315e9e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f38193909830512f070f02c0019a55a4
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8075cb500688c9045232e0142afb660f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c7520a0add26c7ec0df49fd13cd683f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbf851e1195bc3f02e9a67eac9420a4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f16149f870fa5a4c5a8f87641a3e0e7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e29047ffdcd3a8975851024c426cc37c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55018ca3c06d2d90584e4ad45304cdf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dadbab0d21fd5def7720b2b144a083f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_118549b559d1818cf8b38191066565a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 81], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f12ff7ca3a55488a427ae71de17d25a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118549b559d1818cf8b38191066565a2
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_263d7a34ca1065aa378778e3fdabf4c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([1.8156734704971313, 2.0851945877075195, 2.0087790489196777, 2.175837755203247, 2.072892189025879, 2.226891279220581, 2.2403757572174072, 2.028991222381592, 1.9627472162246704, 2.1859636306762695, 2.071002960205078, 2.0156030654907227, 2.1725573539733887, 1.8901900053024292, 2.30122447013855, 2.2802863121032715], dtype='float32').reshape([16]),
                paddle.to_tensor([0.6253926753997803, 0.5155894160270691, 0.8181114196777344, 0.5579231977462769, 0.9746584892272949, 0.6788683533668518, 0.9904800653457642, 0.5969765186309814, 0.7701771259307861, 0.8957492113113403, 0.6820752024650574, 0.5907605886459351, 0.7641318440437317, 0.8520417213439941, 0.9319855570793152, 0.5559881329536438], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_42752d9423807dcd83e1fdfcd2c8ade4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([1.8358906507492065, 2.0165863037109375, 1.986639380455017, 2.0012130737304688, 2.1091840267181396, 2.1463732719421387, 1.894654393196106, 1.9875061511993408, 2.0823562145233154, 2.1248931884765625, 2.0183653831481934, 2.0908870697021484, 2.0069997310638428, 1.9543657302856445, 1.935641884803772, 2.2911624908447266], dtype='float32').reshape([16]),
                paddle.to_tensor([0.37460729479789734, 0.4844105839729309, 0.181888610124588, 0.44207683205604553, 0.02534153312444687, 0.3211316466331482, 0.009519957937300205, 0.40302348136901855, 0.22982287406921387, 0.10425077378749847, 0.3179247975349426, 0.40923941135406494, 0.2358681559562683, 0.14795830845832825, 0.06801445782184601, 0.4440118670463562], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_b1e51bcc452a2eb745c327afbd5f4ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4558117389678955, 0.5129899978637695, 0.5011880397796631, 0.5246601104736328, 0.5184529423713684, 0.5502585768699646, 0.5592711567878723, 0.5030679106712341, 0.4975590109825134, 0.5448992252349854, 0.5135670304298401, 0.511603057384491, 0.5333768725395203, 0.4749213457107544, 0.5690898895263672, 0.5712788701057434], dtype='float32').reshape([16]),
                paddle.to_tensor([0.42562201619148254, 0.3016831576824188, 0.2649511396884918, 0.48494216799736023, 0.07584431022405624, 0.2885129749774933, 0.08532002568244934, 0.48674431443214417, 0.20337152481079102, 0.09332582354545593, 0.03715044632554054, 0.47536739706993103, 0.3538185656070709, 0.22236795723438263, 0.07157617062330246, 0.08841179311275482], dtype='float32').reshape([16]),
            ]


    
    class PrimitiveOp_7b57abf9ba19354a801582580d3c24d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92d185e32e9a08ca1aca5576d6fc63a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b57abf9ba19354a801582580d3c24d7
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_930779e6e28bb805b2d1472eba1816fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.38461145758628845], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_7f75fc64fc005b783511c14608b37c46(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 64, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_056b57ed2f7f5edb9036b26b7ebd2d36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f75fc64fc005b783511c14608b37c46
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_019f417de3583b2dade6fbdcc43c0fec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.09994722902774811], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_402f8917fefa33ac72440a6d78d59063(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 300, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d720fbd2acd9a8cee3ffc2f4942d2843(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_402f8917fefa33ac72440a6d78d59063
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.044002752751111984, 0.4828374981880188, 0.2906877398490906, 0.013120715506374836]]], dtype='float32').reshape([1, 1, 4]),
            ]


    
    class PrimitiveOp_b3f968d5707ed049f5c18e59a2c989e0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea3e2f9bd282297b196738f4f5cc8cd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3f968d5707ed049f5c18e59a2c989e0
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_965846af8aac35995512ae35f1dfccd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3522210717201233], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c6e7648202d253b6a91ccdff005dffa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.477573424577713], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c00733bd785c42a9356135735aebffa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.19731079041957855], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebb74d968f332dddfa74c5adceb663e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_76102f0ee503e6a6ebbe55d0918e46f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eda38591894bd1a54f086ecfd554b135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76102f0ee503e6a6ebbe55d0918e46f4
        def get_inputs(self):
            return [
                paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.21855682134628296]], [[0.10699958354234695]], [[0.44307953119277954]]], dtype='float32').reshape([3, 1, 1]),
            ]


    
    class PrimitiveOp_36b39d8488fb138c1e6aa22ba17a4466(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_100bea0e8c63eda9a3ad9dcf08e941f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36b39d8488fb138c1e6aa22ba17a4466
        def get_inputs(self):
            return [
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a201b12b97258b113ab17ad5195a1fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46fca56dffb5b09f51be872e1cdff4ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c448bc0dd0aacad252e82f230b6360e3
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5c2beaf3386cd8866d12f617452b801(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8d9963ce5ea459f396c97cabe39706e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.04578159749507904], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c856c650e671fd48ff60cf0a187ddf1e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 128, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74a21748569608b31af0e2a5c32d0e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c856c650e671fd48ff60cf0a187ddf1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c29b26bc894030c5d8452e22ce39f01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d0e1374045b258b4193199303cffa70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_873a597832d3331358dc5785c3ab81b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 320, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7052b99a71bbdf384855a5e46560aaf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_873a597832d3331358dc5785c3ab81b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ecfd3a0adc0609c2f664a87f8dbe1f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_419f097a42386cb3ec8d9098edc5ca6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 872, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a70e60484a7118d86d63d966b05b14f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_419f097a42386cb3ec8d9098edc5ca6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98bf7ee0754a8e0271aec4f409c2a142(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 100, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c615d9b3b1d1d4c8aa320b01ad71b5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98bf7ee0754a8e0271aec4f409c2a142
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3af5c0e5de28a1cd21d4eea66ea9f265(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c7520a0add26c7ec0df49fd13cd683f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f9a9b237d379df78c165a9934a6f6a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9a9b237d379df78c165a9934a6f6a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9a9b237d379df78c165a9934a6f6a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9a9b237d379df78c165a9934a6f6a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9a9b237d379df78c165a9934a6f6a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_94c75c085343e2de1204e347bb002976(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20cbb7b4de986160021c03309188525a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20cbb7b4de986160021c03309188525a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9a9b237d379df78c165a9934a6f6a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acbc05f57052ed7e3969284d6ba704a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fa844abfa2ec557e17901738fddb298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_2768523a9f3bb00203021f2dc9af87bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.15479785203933716], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9ba63cdd86268e28a4c5d45dd1d60265(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_109a9133d85c534a9f312d53923f6495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    
    class PrimitiveOp_3f8bd913816f80b5ac1570dcda20179a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 256, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef230714f59728635573639844d5c41c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f8bd913816f80b5ac1570dcda20179a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c7dd7a0b5f07f3e8a27fd70491d4acc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3938605785369873], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_0855e7a1b819acf4eb49f2d9f36e920d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 4, 1, 49, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e68a7439f795a2797139213431e6856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0855e7a1b819acf4eb49f2d9f36e920d
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be15b3a9d946a0a725b4acb45e21ef6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8249001129e909ff8b88def1e823c5f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8656438049c1a0842dfa1a56c09a660
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1ddebf2f0a52c233f6a2ba625eb4262(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92a900e5ee7d41eaf6434b92f08e2e14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8fa295c28651efc9fdac0eebf2afbca6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63845746e785fe152e7001a4c1fe8c11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa295c28651efc9fdac0eebf2afbca6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f57a46db6dd2f675be30795ed390bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5cf05b10f46b186627a0ffebfa9748d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_873a597832d3331358dc5785c3ab81b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a65fe41f351fa787f79424a0dd718ea8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a9c8e25a35b6dd5b555baa2c365eb70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db966dbf3a92120c062087dc9cb665dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6fc139bdd5b3d226683f1ba301088af5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22139849b7d4a56837550142249184a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fc139bdd5b3d226683f1ba301088af5
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6341d2f169b55d1f43855f78f7f839c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.21154245734214783], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.25445711612701416], [0.21916604042053223], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_8741fb547475a6408512aa977372d495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.009391963481903076], [0.19519536197185516], [0.21154245734214783], [0.08582274615764618], [0.24007847905158997], [0.2587222456932068], [-0.2994767725467682], [0.45847052335739136], [0.17372050881385803]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.02195553481578827], [0.15992991626262665], [-0.21389555931091309], [0.14502179622650146], [-0.008672813884913921], [0.25445711612701416], [0.2555086314678192], [0.13578519225120544], [0.3360256254673004]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_169b4ec67da56b44d3c0ae062392507f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.16356325149536133], [-0.1396092176437378], [0.4566301703453064], [-0.2960786819458008], [-0.07350330054759979], [-0.09332668781280518], [-0.0065939947962760925], [-0.2193688452243805], [-0.13021743297576904]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.04090815782546997], [-0.14576321840286255], [-0.1428469866514206], [-0.11748522520065308], [0.3009539842605591], [0.2908601760864258], [0.3797423243522644], [-0.4335433840751648], [-0.18733468651771545]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_3c56ddbfb092e15f94e018b5b60c9806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08131581544876099], [0.19519536197185516], [0.4566301703453064], [0.08582274615764618], [0.24007847905158997], [0.2587222456932068], [0.07948271930217743], [0.45847052335739136], [0.28396111726760864]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31203949451446533], [0.15992991626262665], [-0.09310540556907654], [0.25870460271835327], [0.46219155192375183], [0.2908601760864258], [0.4160849153995514], [0.13578519225120544], [0.3360256254673004]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_567f4d3b69e14da065d747e662959576(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9c2adec2b173a2b00bc78f1bcb0817c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b57abf9ba19354a801582580d3c24d7
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ac3071743993b80f62147fdaaefa251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e18541c18515cf243d858ff7d7df5c65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25899b8d64c96b98380b140b19d50338(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a1cd420d046be81d9368d11da27a448
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fbfd641430878afaba6f85674075bd88(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c0c8ad0df030c4d5af4ce0ecbd69d3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4731403589248657], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9c0c8ad0df030c4d5af4ce0ecbd69d3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4731403589248657], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_4d97c383c90a67344e00ce6c3ca26500(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8c462a289dd4b02a837cbe5adcb4511(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d97c383c90a67344e00ce6c3ca26500
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0d8e766e44103241e3660819e7afd21(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc8eb7e2d03650405791270e961d96b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0d8e766e44103241e3660819e7afd21
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.37376904487609863], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d6039a7e0644686d3a4747dbffb69099(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbd624b2f425754566721f11b1e8945b
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8217e1212906715377f99595ccb872b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a1cd420d046be81d9368d11da27a448
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf679d53c59eba522a8b1c76334c8f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_077bdda4e0d5f7b56b46693a2b10f23b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_077bdda4e0d5f7b56b46693a2b10f23b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_077bdda4e0d5f7b56b46693a2b10f23b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_077bdda4e0d5f7b56b46693a2b10f23b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_077bdda4e0d5f7b56b46693a2b10f23b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1546777cce0c28c4626a1bbc7e0138e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1546777cce0c28c4626a1bbc7e0138e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_077bdda4e0d5f7b56b46693a2b10f23b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8133b005c68e26031574f74885ff1996(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_58510909880ed13355009b3c9250e1e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8133b005c68e26031574f74885ff1996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_374f49606609c2dc0f6d71699b2c4302(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a1cd420d046be81d9368d11da27a448
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a21f2824a98bd2e82fef553f3a0b19d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a16bff1fbe607c714b72b5634de63526(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a16bff1fbe607c714b72b5634de63526(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a16bff1fbe607c714b72b5634de63526(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f0a78eaa9c3d9f8d1fd939044d00564(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51689daa60cfc0078faf6cab39d17e61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.33328139781951904], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ff6a99261bff06fb07f371e3793c55fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb680be637ebf862893d6c10a0271db0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38a20666206564ec76d6e5b6555f8886(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcd94a7f4609143b0012f8929f5ae27e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8656438049c1a0842dfa1a56c09a660
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e0a0116bc4cec67ef88ceb6ea860220(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1028a869fa345bd4410b3ef7d712fe60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_69c4c8ba427e24576294e27d131e1ab5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1028a869fa345bd4410b3ef7d712fe60
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f4b8c8f3f5076cd185173465367d759(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0675347dda13a41bc217256d9c50a6e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118549b559d1818cf8b38191066565a2
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e4ea347a8d5e498c3419defe83b2e01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4609eab220565b37c5d0d048f837660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3866c5399fcb7d583a258b6f5949f72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6a1b1dde356d362ce5d05c90ac9cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.31139373779296875], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4757fd7c02a4372410c30a3f069b8d2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9dc7b596d09fbaacda68bea8f7c0fe7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.8605836033821106], dtype='float32').reshape([1]),
                paddle.to_tensor([0.3450847864151001], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_070cd1b7c3580c1c08d2312295219efa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6908892393112183], dtype='float32').reshape([1]),
                paddle.to_tensor([0.02924232929944992], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_dcbfa7479327897b5258bd3eb34e7449(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 1, 49, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d982d7c8daf6092692ed6a102cac48a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcbfa7479327897b5258bd3eb34e7449
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_396081d327ad7011cbde4dad47072200(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6806e728dca212338f9cfe06cc585eca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc4d6f6d852e0a9729edd5248e83e90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55644d8b93ae5604d11455d1409e3cc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e065d965f80cfe61d4cc5f58f8d16f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66701d69fa04609e8206edd57514a2dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0bca5aa3a098abebebfefe6a30f0aafe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 36, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c03ef38c7b786a6c65edfb2d58134049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bca5aa3a098abebebfefe6a30f0aafe
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5290ac62bf3914d9b646ba395782a08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12313771992921829], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fe916e29c7e10c5c102f97cad3eb9ff5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32f6e61a35b886e78ef1f586386a6362(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12150554358959198], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_46234399daf76739a79e60066983e99e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22139849b7d4a56837550142249184a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fc139bdd5b3d226683f1ba301088af5
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a44b1b3c666f374b5c4b73304de5ab5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b232a8379f3b93c0687016d235139e8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10905a55e0a8ce1415fc6a8332cb9596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2766314744949341, -0.07481521368026733, -0.03476780652999878, -0.2757680416107178, 0.0, 0.02061089128255844], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, 0.1621771603822708, -0.20417125523090363, -0.3648090064525604, -0.0037841498851776123, -0.3224535882472992], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_50fbb131c90b9b45ff3d00dae13da0a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, -0.012133318930864334, 0.007098586764186621, 0.1006026640534401, -0.0, -0.006646055728197098], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_11a40ea4ae7719990bd6ab8b4f233154(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, -0.0, 0.0, 0.0, -0.0, -0.006646055728197098], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_bae564c12bcbfb2080ca48796a874d99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.21817441284656525, 0.0, 0.0, 0.05182693898677826, 0.0, 0.02061089128255844], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, 0.42682701349258423, 0.020339012145996094, 0.0, 0.0, 0.02947470359504223], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_3473af602531c81d140c934efdeb3e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2766314744949341, 0.10538440942764282, 0.14060714840888977, -0.007649481296539307, 0.0995698869228363, 0.10518306493759155], dtype='float32').reshape([6]),
                paddle.to_tensor([0.09058192372322083, 0.1621771603822708, 0.031248152256011963, 0.106215700507164, 0.17448961734771729, 0.0300922691822052], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_741480208c0362158e60cea4e623c850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0168093740940094, -0.15478301048278809, -0.01066252589225769, -0.29785677790641785, -0.09646612405776978, 0.032794758677482605], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0168093740940094, -0.15478301048278809, -0.01066252589225769, -0.29785677790641785, -0.09646612405776978, 0.032794758677482605], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_efe8c5765fa30340be76cf124e5574e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.053348422050476074, 0.0016325414180755615, -0.22996483743190765, 0.2222844958305359, 0.0067261457443237305, -0.35223710536956787], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.053348422050476074, 0.0016325414180755615, -0.22996483743190765, 0.2222844958305359, 0.0067261457443237305, -0.35223710536956787], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c640aa10d6149834fa01563ad2792fd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.21817441284656525, 0.18019962310791016, 0.17537495493888855, 0.3199455142021179, 0.0995698869228363, 0.10518306493759155], dtype='float32').reshape([6]),
                paddle.to_tensor([0.21817441284656525, 0.18019962310791016, 0.17537495493888855, 0.3199455142021179, 0.0995698869228363, 0.10518306493759155], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0e989e72105bbcca227c75e6d34aef52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09058192372322083, 0.42682701349258423, 0.2557584047317505, 0.47102469205856323, 0.1782737672328949, 0.3820205628871918], dtype='float32').reshape([6]),
                paddle.to_tensor([0.09058192372322083, 0.42682701349258423, 0.2557584047317505, 0.47102469205856323, 0.1782737672328949, 0.3820205628871918], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_52ea9610077e311ad3d94a938bbf41df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0340176522731781, 0.28506091237068176, 1.14909029006958, 0.024235691875219345, -0.05860188230872154, 0.2763666808605194], dtype='float32').reshape([6]),
                paddle.to_tensor([0.08393514156341553, 0.7033591270446777, 2.8352646827697754, 0.059799134731292725, -0.1445942521095276, 0.6819069981575012], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8b46e7b2839d0c4ab43bd38b6f520c68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0028471469413489103, 0.16701386868953705, 0.7651466131210327, 0.001447176095098257, 0.008402298204600811, 0.15857239067554474], dtype='float32').reshape([6]),
                paddle.to_tensor([0.002855276456102729, 0.20050019025802612, 3.2579751014709473, 0.0014492734335362911, 0.00847349502146244, 0.18845637142658234], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d38e2015b17aa4a7bdadc4d42ea6fd86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f72ee08bd49926f5a12dc9ea61f4d8a
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95c397429bf17c064737d44b8e17b8b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c856c650e671fd48ff60cf0a187ddf1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebf8b325fd88b44d9538be10b60f920e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d3297a525836585cd0588b4c50ec917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d3297a525836585cd0588b4c50ec917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d3297a525836585cd0588b4c50ec917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d3297a525836585cd0588b4c50ec917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d3297a525836585cd0588b4c50ec917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d60d15c272fae8d00a73ed096e18dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d60d15c272fae8d00a73ed096e18dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d3297a525836585cd0588b4c50ec917(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf679d53c59eba522a8b1c76334c8f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85c6d162df54e46e65ab0c79f3d347f7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4a5b9011077aa2009faff34261e1b7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85c6d162df54e46e65ab0c79f3d347f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9445b1fdf1aa795a9556b20e366c976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85c6d162df54e46e65ab0c79f3d347f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b764c635369fada903cc209607ba376f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8656438049c1a0842dfa1a56c09a660
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acbc05f57052ed7e3969284d6ba704a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ace8dfff6cef94fc1aa3d2755568558b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a1d990179688d9253d7d9521a5509f6
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efbd4b51b8db1a358ec874faa4052fa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bd706db41e7dc8fbdf93f89d74d6e38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4205133616924286], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_0d45bec6fe8ad7f77d6bb7542bc45c20(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3267163f84119ff3d8b4c713b4030e43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d45bec6fe8ad7f77d6bb7542bc45c20
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb6ade180205598ab74ab63738e77710(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6890b78be95ff68cf0bfd348da9440c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fc139bdd5b3d226683f1ba301088af5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c28a5ad00075609c600bee8fc14ca140(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbd624b2f425754566721f11b1e8945b
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_519fa81a8851335a6b6fe5de08847567(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b57abf9ba19354a801582580d3c24d7
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_464826b394ce2b3137984d637036871f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8656438049c1a0842dfa1a56c09a660
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9270f7f853ee2669d437e7f25a0878c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eae590300c0334574b7c70af63054e67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_16dccfb4b00180ff3e6465c3d457b873(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eae590300c0334574b7c70af63054e67
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76b5014324887bbfc22dc92aef9f4734(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e79224fd84a1b99578b31c6269aff669(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.30836474895477295], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_e7fcd4e5c69c0c4301e973cc46ad64f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 8, 1, 49, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45338e76c1ffc28c9d44ac98280e6936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7fcd4e5c69c0c4301e973cc46ad64f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e60e9b8fddce4848be7b48b2cead759(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([2.0022754669189453, 2.240567207336426, 2.111360788345337, 2.1870291233062744, 1.9671711921691895, 2.0732555389404297, 1.9710893630981445, 2.024165153503418, 2.334258794784546, 2.063455581665039, 1.9864293336868286, 2.0223824977874756, 2.1271204948425293, 2.148427963256836, 1.9668179750442505, 1.997074842453003, 2.1886327266693115, 2.188886880874634, 2.300858497619629, 1.8540812730789185, 1.9147701263427734, 1.975813627243042, 2.1764822006225586, 1.938555359840393], dtype='float32').reshape([24]),
                paddle.to_tensor([0.8168052434921265, 0.7831318378448486, 0.5267834663391113, 0.96436607837677, 0.7379273176193237, 0.869096040725708, 0.5500746965408325, 0.9029768705368042, 0.5807815194129944, 0.6169620752334595, 0.5883928537368774, 0.7845883965492249, 0.8679076433181763, 0.5795742273330688, 0.5975856184959412, 0.8808311820030212, 0.8277290463447571, 0.6553834676742554, 0.5039186477661133, 0.8484286069869995, 0.8710693120956421, 0.6950018405914307, 0.995990514755249, 0.5658894181251526], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_309c333c6cb3f4fe8817a77f77429087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([2.219653606414795, 1.9027743339538574, 1.973353624343872, 2.1888537406921387, 2.2433595657348633, 2.0183637142181396, 1.9209812879562378, 1.9374973773956299, 2.027986764907837, 2.10695219039917, 2.2992305755615234, 2.2776315212249756, 2.1760663986206055, 1.852568507194519, 2.2449796199798584, 2.340653657913208, 2.0092720985412598, 2.016832113265991, 2.133084535598755, 2.299717903137207, 2.0900368690490723, 2.204993724822998, 1.8239502906799316, 1.931168556213379], dtype='float32').reshape([24]),
                paddle.to_tensor([0.18319472670555115, 0.21686814725399017, 0.4732165038585663, 0.03563392162322998, 0.26207268238067627, 0.130903959274292, 0.44992533326148987, 0.0970231369137764, 0.4192184805870056, 0.3830379247665405, 0.41160717606544495, 0.21541158854961395, 0.13209235668182373, 0.42042574286460876, 0.40241438150405884, 0.11916881799697876, 0.17227095365524292, 0.34461653232574463, 0.4960813820362091, 0.1515713632106781, 0.1289307177066803, 0.3049981892108917, 0.004009498283267021, 0.4341105818748474], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_89b8dd68e2fa7316eac740f19805f0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5105245113372803, 0.5418276786804199, 0.511513352394104, 0.5467735528945923, 0.5098881721496582, 0.5165175199508667, 0.4871361255645752, 0.5039390921592712, 0.5514659881591797, 0.5200290679931641, 0.5287951231002808, 0.5193415284156799, 0.5333964824676514, 0.5060102343559265, 0.5196885466575623, 0.5095046758651733, 0.5394335389137268, 0.5323984622955322, 0.5544072389602661, 0.4804067611694336, 0.4843418598175049, 0.5114282965660095, 0.5437671542167664, 0.48383718729019165], dtype='float32').reshape([24]),
                paddle.to_tensor([0.43111735582351685, 0.2893909513950348, 0.15826816856861115, 0.4119057357311249, 0.24751590192317963, 0.3494945168495178, 0.3399348258972168, 0.38630738854408264, 0.3234490156173706, 0.09757565706968307, 0.2129882127046585, 0.46143293380737305, 0.20221589505672455, 0.08494094014167786, 0.43189936876296997, 0.3254511058330536, 0.13017435371875763, 0.2705845236778259, 0.48569831252098083, 0.06358955800533295, 0.33293449878692627, 0.40925174951553345, 0.0861608162522316, 0.37648701667785645], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_d7dda93ebdda52e063c81e7b800a1b5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c448bc0dd0aacad252e82f230b6360e3
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_423c8da05f45dc956651bccf28acb735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_423c8da05f45dc956651bccf28acb735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_423c8da05f45dc956651bccf28acb735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_423c8da05f45dc956651bccf28acb735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_423c8da05f45dc956651bccf28acb735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6267cb1a510215a9b970ecc581227b08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6267cb1a510215a9b970ecc581227b08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_423c8da05f45dc956651bccf28acb735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_670284e0697f88728dc5e83d341bfc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_670284e0697f88728dc5e83d341bfc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d62aa88c5e5fcf00859a56ca47906dc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5a4f5be8ada73ef594e3aeb38ea4d94
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24634625017642975], [0.2473631501197815]]], dtype='float32').reshape([1, 2, 1]),
            ]


    
    class PrimitiveOp_892c57dc68061d7c5aa6e1f72e71f53d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3549, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e892b0181638b57ee6fffdbe2571dde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_892c57dc68061d7c5aa6e1f72e71f53d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ffdbba15f191a4fdacfe4a8920b086f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 16, 1, 49, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_674d8cf4ec0ce6c840f1972a5f24bfc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ffdbba15f191a4fdacfe4a8920b086f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f5792c689efe13eb2b93b3c07c99f53d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 288, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1056bac98ce317e7d85839490bd56a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5792c689efe13eb2b93b3c07c99f53d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e0b8a3c0f6aaf1dff5c3bd9a25320fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5facc5d8413cd57b9ce20fbff4034eca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30a782a51c0fbf7e74ee53625f56816d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b57abf9ba19354a801582580d3c24d7
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f97881b365064cffa2800c3ce7ba8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12146636843681335], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b76f7c3790cdf130924c9528fa8f9f24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0e27d127f402ef84158c95622351211(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e930ca8185afe2dea3f9a8781fdcd9aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([2.0273215770721436, 2.051156520843506, 1.8777961730957031, 2.1210427284240723], dtype='float32').reshape([4]),
                paddle.to_tensor([0.9491412043571472, 0.7706915140151978, 0.8685111999511719, 0.8350591659545898], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_ace8d5003d862f0bd176a7dea39a34f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([1.9933178424835205, 2.198335647583008, 2.116931438446045, 2.207944393157959], dtype='float32').reshape([4]),
                paddle.to_tensor([0.05085877701640129, 0.22930851578712463, 0.13148881494998932, 0.16494080424308777], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_2998c4efe4c71424b836428fcef81d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5063980221748352, 0.5212265253067017, 0.4773099422454834, 0.5338440537452698], dtype='float32').reshape([4]),
                paddle.to_tensor([0.493753045797348, 0.14767983555793762, 0.4444725215435028, 0.12316848337650299], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_248f13d13207a80102e14933d66b2199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0029604582a10483549d3f1c94229675
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_990eef402fb5bab1fa44c1b53384f5b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a2c6da5b7d2d9cba68c31dbdf54f087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f75fc64fc005b783511c14608b37c46
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8909fb57d09a05fe23cc9d2e223d7c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b232a8379f3b93c0687016d235139e8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecba227fd86fe3dc0c33e8f069794fc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bca5aa3a098abebebfefe6a30f0aafe
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8904fdb688cc4a664b2ea8afaa750a2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9b3e45d958d65bcfd00836212380185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8904fdb688cc4a664b2ea8afaa750a2a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2909002900123596], dtype='float32').reshape([1]),
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce21b5a12358cb117bdf05ac28ba418a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07224521040916443]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_be91cb14cbf4db5aaee9cf1c76ffcf72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09675587713718414]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.04453441500663757]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_260772738db026605ac0d6fa50aa3687(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10698825120925903]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.14122912287712097]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_6af37a2b9ab34ce8f5429412a4d9c113(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13149891793727875]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.19459663331508636]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_674d8cf4ec0ce6c840f1972a5f24bfc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ffdbba15f191a4fdacfe4a8920b086f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4cbbc94be80c5c26af3114cc037bd02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.11981570720672607], [0.14607250690460205], [0.1565159410238266], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d23e1e6d4b915830484f5d4b6792e6f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08189225196838379], [-0.1403680443763733], [0.07632625102996826], [0.03819593787193298], [-0.01861479878425598], [-0.09420964121818542]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.1627451777458191], [0.25780361890792847], [0.2799569368362427], [0.068025141954422], [0.0581510066986084], [0.08422863483428955]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d57996aab73a74e636b5cdf391eee717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.1917412281036377], [0.039921388030052185], [0.13284483551979065], [-0.3065911531448364], [-0.05574962496757507], [0.30769357085227966]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.1562783122062683], [0.20795682072639465], [0.29048770666122437], [-0.0940697193145752], [-0.0030398517847061157], [-0.08507192134857178]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_9868fbf45f195a32b3e16aef8564d185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01215440034866333], [0.039921388030052185], [0.3015764653682709], [0.0914648175239563], [0.2659372091293335], [0.30769357085227966]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.19920778274536133], [0.31968793272972107], [0.41392871737480164], [0.08479256927967072], [0.21667928993701935], [0.09567250311374664]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_87352f6f6f7694a8487dde5469891bf5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2699f2bd524822d2abeb24b86240a8ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ce202c0830b0253513bf785c7e58af6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeb585d2abe3bd8cde79418c435e8ca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_8a0c78d059f6bbb858858033446d06da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0ced8f919661f77f95297c531fd7fb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c50bf0331aeaa47dfe0cc8b1b15c918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480c7d6d76b1578ad94bdef5a8b5a915
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.7649943828582764]], [[0.7536299228668213]], [[0.8622885942459106]], [[0.8777257204055786]], [[0.7248691916465759]], [[0.7289198637008667]], [[0.8996758460998535]], [[0.8994820713996887]], [[0.9005266427993774]], [[0.8886114358901978]], [[0.8354483246803284]], [[0.8511449694633484]], [[0.8733361959457397]], [[0.9242442846298218]], [[0.8901932239532471]], [[0.7633470296859741]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_e89eced0f0a429ecf06191e0083592c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6102a96bbfac6bcbd78ff5f3e9ebcddb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fe46bd115631ded35c76a3f4c62756c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f8bd913816f80b5ac1570dcda20179a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a7de8c9c59dd6f3de11662847fdf227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c448bc0dd0aacad252e82f230b6360e3
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d05627ff5da69f88116aea2e8af7e417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.34778791666030884], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5467f470ca821827ed1d62413fdbf941(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118549b559d1818cf8b38191066565a2
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6bf38c2d1d71b091cc340168ca57d463(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48bb70cf1ed703e6d3564092edf7b164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bf38c2d1d71b091cc340168ca57d463
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8514b99b9f1fe5a4efcaa9a34833bcd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8514b99b9f1fe5a4efcaa9a34833bcd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c41b1bc89fe48c76b22ef8015880cf3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5a4f5be8ada73ef594e3aeb38ea4d94
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24644309282302856]]], dtype='float32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_a6e795331cef01bed51f93a59b3d6c7d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4116, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10da5ddda354ab474573eaf6f6afbf7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e795331cef01bed51f93a59b3d6c7d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8436018583bec4dc9c2cf4eea892ee71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4dc3a52051b0916b7ba8341b54baf69c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 400, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 400, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bbaf13537936f7996a26f96f6c824877(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dc3a52051b0916b7ba8341b54baf69c
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ad35e3efd5ad67cdeae7bcb3d6889e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8656438049c1a0842dfa1a56c09a660
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2de205c4f6178cc0caec0f305894fc1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_461bc1e31ac89a83ed02394249f48fd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc6643e5b2bc39746a15eaf6583e90d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c7520a0add26c7ec0df49fd13cd683f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c13d82e482be2c001f3a19289203b501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5ed9377b9307d73648a7a22011a73af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cf0fd2c2812e68b666d6043426a84be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5625240b608b8e179afd6b46d021ccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173ec1c225696f5b99f43aeb425ffd75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa295c28651efc9fdac0eebf2afbca6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_16fbfdcc49bf0f4b3864926bf5be4738(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 336, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 336, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37e57138e1424efa348a6c7ff21f7bd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16fbfdcc49bf0f4b3864926bf5be4738
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a0afb548beb0476416685197b17d911(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e409628fc292e5ac00a418913fd15879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b56b6af65c31b1e021422b1e7649e66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d45bec6fe8ad7f77d6bb7542bc45c20
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45338e76c1ffc28c9d44ac98280e6936(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7fcd4e5c69c0c4301e973cc46ad64f4
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ca84d26b459637197c1f6232a7e4396(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14e2a78cb1b09cf9545cdf70e223df84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dc642d2c841205a0d31239189615084(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa295c28651efc9fdac0eebf2afbca6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ff4acf90c32bf5e31248b962ee78aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0bd285d471000535c0091c333342a48
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75b8735e43cede6507be85d7c4789a80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ba333df76203ca8b2f4004929798fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dc3a52051b0916b7ba8341b54baf69c
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4a9b7f6cf992ae101901f3451285666(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.40281563997268677], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c27bd5175c8678d603d94a4b661e382d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d7396952c703319f33e88321947bf1f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6234cc38ac5ae83eee72c5beb332a454(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7396952c703319f33e88321947bf1f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_09bcddc38cdd7faa655524e2404ffd16(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 56, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 56, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae741c21bb6ddc3f91c8cb68f2b4304a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09bcddc38cdd7faa655524e2404ffd16
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_693ed7ae37911d435b060aadd39b0ea8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b62cdc358a5dc4bf3fcb5f3f549eef20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b62cdc358a5dc4bf3fcb5f3f549eef20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b62cdc358a5dc4bf3fcb5f3f549eef20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b62cdc358a5dc4bf3fcb5f3f549eef20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b62cdc358a5dc4bf3fcb5f3f549eef20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb63cea6476a7d0b9a67ccc96c2bbd8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb63cea6476a7d0b9a67ccc96c2bbd8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b62cdc358a5dc4bf3fcb5f3f549eef20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2a51ebd81954dedac4844ef20862fb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2c159559d881ff96fcb996034416d9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.9709991812705994], dtype='float32').reshape([1]),
                paddle.to_tensor([0.0655159056186676], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_687e6a3e7265071926a67c672e8f9bed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.9603669047355652], dtype='float32').reshape([1]),
                paddle.to_tensor([0.24741116166114807], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d07c1cdb5311809ce32aaac0e29c0305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6136724352836609], dtype='float32').reshape([1]),
                paddle.to_tensor([0.04495077580213547], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb05d399e7c42daee0e38184c59eede8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6127894520759583], dtype='float32').reshape([1]),
                paddle.to_tensor([0.11855943500995636], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_82cec0939daa313864a8dd810efb32b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6118536591529846], dtype='float32').reshape([1]),
                paddle.to_tensor([0.08503968268632889], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0824fbe855837d21b4f371587ec80bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.9667161107063293], dtype='float32').reshape([1]),
                paddle.to_tensor([0.06937023997306824], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f3706d5c7e467f37c13d7b33d374f9e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6435269713401794], dtype='float32').reshape([1]),
                paddle.to_tensor([0.30397114157676697], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5ad118afb34349b5029550717b8b55da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.7003757953643799], dtype='float32').reshape([1]),
                paddle.to_tensor([0.4753126800060272], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f50d883179806b9ac8643fc88ca66a9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0089ad2f695e76d15a2c1bfa3cdf5673
        def get_inputs(self):
            return [
                paddle.to_tensor([0.7971274256706238], dtype='float32').reshape([1]),
                paddle.to_tensor([0.21060983836650848], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a0b220341afaeded814e2d9a8258e5c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdaf0d76f93adecb4d9c357efe707c8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aed498bbb3d5698a12b7cd744b424c57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d2a34f0cecae9fa17bb84d407272c72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bea8baf18358ee34a4936c414b9e46f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.27113786339759827], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bea8baf18358ee34a4936c414b9e46f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.27113786339759827], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_25fbcf07b04ab2ca0fabe8c885072fc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d97c383c90a67344e00ce6c3ca26500
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_563d602218d5573086b131b08b1dd623(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0d8e766e44103241e3660819e7afd21
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.19904105365276337], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2815235aee7ee6f7e598a4632361c298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0029604582a10483549d3f1c94229675
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a512c9892a1c2c9407c403f85989a450(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa295c28651efc9fdac0eebf2afbca6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f23c2c883695f955d58e9d972a83ed29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f23c2c883695f955d58e9d972a83ed29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f23c2c883695f955d58e9d972a83ed29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f23c2c883695f955d58e9d972a83ed29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f23c2c883695f955d58e9d972a83ed29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cc69ec42413e9683354fab45aa84741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cc69ec42413e9683354fab45aa84741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f23c2c883695f955d58e9d972a83ed29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6806e728dca212338f9cfe06cc585eca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7aa7bc4d853eb930aeac36e5d55f0c35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ecb49b8fe72c4d51bdb18029ca6c71c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aa7bc4d853eb930aeac36e5d55f0c35
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.4964856207370758]], [[0.40626269578933716]], [[0.46424606442451477]], [[0.3560028672218323]], [[0.10537204146385193]], [[0.3426622450351715]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_47a8e1a8eefcba7cc0523bc0d875f244(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_669ceb924168c8e3c7d3aff893984901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_669ceb924168c8e3c7d3aff893984901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_669ceb924168c8e3c7d3aff893984901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_669ceb924168c8e3c7d3aff893984901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_669ceb924168c8e3c7d3aff893984901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70964aea4657b22c07dc187cce70b126(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70964aea4657b22c07dc187cce70b126(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_669ceb924168c8e3c7d3aff893984901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e7e466e11cf0fce830e877508357a65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231d7fcaa6e7ec7976e960da177ad71e
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d2d386b5f35c2f0cb5df85f9b6dacd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17c24753cf144df5d233443d4d7ef633(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d16a652b2adf2ce668ca961d1da7b39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb979fc82f9e197210fa30d4fa045c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d6676265940b4d11cd20af766da8c0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 24, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_efa58ec9b9dbfe042ab81472e93b0e76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6676265940b4d11cd20af766da8c0c
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7599f56b2b4f4c541657a0b5e8d0d627(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6676265940b4d11cd20af766da8c0c
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fbf21c4f12d6e291b9c321d16b1b37a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6676265940b4d11cd20af766da8c0c
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_753c013786351cb50083eecbda1df44b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6676265940b4d11cd20af766da8c0c
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ace8dfff6cef94fc1aa3d2755568558b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a1d990179688d9253d7d9521a5509f6
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0787337fc9d78fe7ffa109369a0c3e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ce202c0830b0253513bf785c7e58af6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9c8059a92b736be026b879140319e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_639ecb4d17fc7a2dc54bb7eea87c8e5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f58222d1f0d4f6229a08ba7643ded332(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8575fe42e0a8b730d655ac33dcb7281d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8575fe42e0a8b730d655ac33dcb7281d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aafdd7085ea0fa2588bd55892d00d0d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.48570457100868225], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_457dc89274e6f485ff075b80a0e8aa8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_62266086d59a19263d5aa2a242ad94b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 200, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 200, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a6eb1cca328fa5aad292644cb87634d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62266086d59a19263d5aa2a242ad94b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_247af07ceeac1785a65728f7a18e749f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e2a7e971c8415121d8e59282c480f9b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 1, 49, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0053bb80bc501a5d5b2bfe7d3efd102b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2a7e971c8415121d8e59282c480f9b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58510909880ed13355009b3c9250e1e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8133b005c68e26031574f74885ff1996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58510909880ed13355009b3c9250e1e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8133b005c68e26031574f74885ff1996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58510909880ed13355009b3c9250e1e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8133b005c68e26031574f74885ff1996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef44f7e1827a4e0c4bdab9c1fda99b56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b4a8cad0e8cadb8e50a154ba5c1a6eae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef44f7e1827a4e0c4bdab9c1fda99b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e55da620b0795f6890977c83e6b5f17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eefba6d6dd56eb4581ba8639585abea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eefba6d6dd56eb4581ba8639585abea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fee4d718992b4a1dd9f6f71de646908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.09488201141357422], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_91ee3935b0dd9c4678d436d4fb94fd0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06572425365447998], [-0.017537042498588562], [-0.16134634613990784], [0.1909516602754593], [-0.2708263397216797]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.2624059319496155], [-0.25955161452293396], [-0.3877258896827698], [-0.1352548599243164], [-0.3994396924972534]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_73b566af1e0dfb5395939469b1d7003c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.02962172031402588], [-0.012923628091812134], [-0.3421880006790161], [0.09488201141357422], [-0.03534042835235596]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1962345689535141], [-0.3319106101989746], [-0.016855984926223755], [0.2746593654155731], [-0.3117551803588867]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_a6a2951b70cf9e4e54bc43644c6c6a86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.022850245237350464], [0.2385021597146988], [-0.08091486990451813], [0.1909516602754593], [-0.03534042835235596]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1962345689535141], [-0.25955161452293396], [-0.016855984926223755], [0.30179738998413086], [-0.3117551803588867]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c10c92c1b6dabb464ca45c97189f9745(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c854bbd89649f27ac58f956da786e13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_807156cfdda5be1ee89612ef4f861183(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cf3343ba9b9329e68e46897656e4ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d37c59f56bf6b74141893902d22eb3aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_446b9da22f79069b8a9377cb5cdde2f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52e893622f7c7b6f125c67fe610db09c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_446b9da22f79069b8a9377cb5cdde2f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4739fd4acd92ca666d6947fb0cf4a775(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ec38bfc95a9a48f223f6f8f1911beb2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2aa9df2ef41a3e1535e8a5ae6284be3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3024537265300751], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_62b7e86bae7e25778160673d0855764d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98bf7ee0754a8e0271aec4f409c2a142
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e2af1ea5fad46f0074d4dc27a9ca3f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b86efe3698a902796d42337820c9e7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5792c689efe13eb2b93b3c07c99f53d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1e0d4b4ac7b455f1825f5121648c58f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72a6b511c486ee4581cf227550d83e70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbd624b2f425754566721f11b1e8945b
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_584760a1acce6526514acc3f019b8a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8904fdb688cc4a664b2ea8afaa750a2a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3566078841686249], dtype='float32').reshape([1]),
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21434a7cdb5255da06d1948e83a04f81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8133b005c68e26031574f74885ff1996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21434a7cdb5255da06d1948e83a04f81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8133b005c68e26031574f74885ff1996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21434a7cdb5255da06d1948e83a04f81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8133b005c68e26031574f74885ff1996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0344151b99a5168075323f3874738f46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef44f7e1827a4e0c4bdab9c1fda99b56
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ae2d0213313a93bdc739f8eefe7549a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1248, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc1785126f7142f2d9795b5f3aeb33bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae2d0213313a93bdc739f8eefe7549a0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9db85e3cc97692eb8a55a238b83c333(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5792c689efe13eb2b93b3c07c99f53d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce2704f934b7c5e65a27cb27dc1d0297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.010624636895954609], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ce2704f934b7c5e65a27cb27dc1d0297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.010624636895954609], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b2c2e4e51c904e667f05f336169229bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d97c383c90a67344e00ce6c3ca26500
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4d32a43abb851561c73a6a6afff3a9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0d8e766e44103241e3660819e7afd21
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3146562874317169], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619a85a8d81ad89bd9481b30204cb082(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f72ee08bd49926f5a12dc9ea61f4d8a
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39d0907ea3f5cd92eb06c4bdea049a4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bca5aa3a098abebebfefe6a30f0aafe
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dc2b513b132e0c21d600ac177762e6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_446b9da22f79069b8a9377cb5cdde2f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51732bfa540973bc7197bd7de15d69b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.37174689769744873], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a406a8741e094507ffd249d76c6409c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1aebda66a7141afe6cacdd21537cf0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2051a5a226cc7365e21b7bca67e63d64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbd624b2f425754566721f11b1e8945b
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e68ed81f1d609fa3055cbeb09c8f294d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e68ed81f1d609fa3055cbeb09c8f294d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e68ed81f1d609fa3055cbeb09c8f294d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e68ed81f1d609fa3055cbeb09c8f294d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e68ed81f1d609fa3055cbeb09c8f294d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7eeefc5fa6a6e4ce11d8b0dcc3be800b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7eeefc5fa6a6e4ce11d8b0dcc3be800b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e68ed81f1d609fa3055cbeb09c8f294d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcb841341637170ce57ea4e164c5f1b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_970eb97ae963c961831ac96a7b754add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0dae384ccd7dd87e9f09a4a00c3364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0dae384ccd7dd87e9f09a4a00c3364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0dae384ccd7dd87e9f09a4a00c3364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0dae384ccd7dd87e9f09a4a00c3364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0dae384ccd7dd87e9f09a4a00c3364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a3b829a377e68b23f20d56ecf4c6ffa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a3b829a377e68b23f20d56ecf4c6ffa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0dae384ccd7dd87e9f09a4a00c3364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac023574f8b779796d2fa45cbc8efdec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac023574f8b779796d2fa45cbc8efdec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac023574f8b779796d2fa45cbc8efdec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac023574f8b779796d2fa45cbc8efdec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac023574f8b779796d2fa45cbc8efdec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8abd7165d9ddb503791a636cbc8dfc8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8abd7165d9ddb503791a636cbc8dfc8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac023574f8b779796d2fa45cbc8efdec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f3c4f4f2d16b3d9979e522a2c66a25f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2106606662273407], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4f3c4f4f2d16b3d9979e522a2c66a25f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2106606662273407], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3ec6c4b58ba8a3aab89fb154d33f32ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d97c383c90a67344e00ce6c3ca26500
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b04b270d79ecfc31d13e06f8d4be53fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0d8e766e44103241e3660819e7afd21
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3481174111366272], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_43eedddf7ad7fa6672c807350f91ad94(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 156, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ea4e574efadaf2d656ca8cbf14fc4f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43eedddf7ad7fa6672c807350f91ad94
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9ac75308f7727d9937f5cf9e09bdbea1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_319ee4d4b5795bd5bc327d12936f2863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ac75308f7727d9937f5cf9e09bdbea1
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.8479311466217041]], [[0.8519147038459778]], [[0.8904101252555847]], [[0.8855916857719421]], [[0.8917824625968933]], [[0.8464255332946777]], [[0.7774280905723572]], [[0.8978089690208435]], [[0.8423903584480286]], [[0.7676694393157959]], [[0.892359733581543]], [[0.8445429801940918]], [[0.8606671094894409]], [[0.7275188565254211]], [[0.8979865908622742]], [[0.8983840346336365]], [[0.8124725222587585]], [[0.8320170640945435]], [[0.8500043749809265]], [[0.9105454683303833]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_fcfe867acc6a6a5ecb8a1820cd0300db(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cdf7d619a050d66ceef360714388a48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcfe867acc6a6a5ecb8a1820cd0300db
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_72615d8c339c44faccc1208b338d8ed1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e34355f1f7c173cc30591f5e31fb383b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72615d8c339c44faccc1208b338d8ed1
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e6f9790d4bc0bfb505b3e1965a8dd7f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bfc06c2d0a81c4224a5b30e35703658c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9790d4bc0bfb505b3e1965a8dd7f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f83ff0f6549b5687f2fca2c7d50dadf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8891e681dcb44fd5b9e1cffc1f609406(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06a0910957185fde1a2a21029d563401(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82be4f61484de9e62614ae96241e8690(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62266086d59a19263d5aa2a242ad94b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de15c788f48972d33bacfad94b85e7a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01c5a7a46f3ea4e42c001962b2f79742(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_4c108cf318bf6701509f465411d12652(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_873a597832d3331358dc5785c3ab81b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90929ba91027117d13341f41211707b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8904fdb688cc4a664b2ea8afaa750a2a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.19080998003482819], dtype='float32').reshape([1]),
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_979fb291e864fbdbc4a99fa05c1a8071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c89639e795f07d6397c59bac331f1e5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db6c3abbaff42a60a1bfc05c709a460c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_291e09dcea47470981b1cf5656085cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d40be0f300ad4d2959f5cb89b1bbc896(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd3aa9e535400b3119989222c0a59f9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_419f097a42386cb3ec8d9098edc5ca6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a62a33ff5c69d19150a4df8859ba64c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0b0d245f498b74481040c112ecd492d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8e9a4e126c6ac141c1dc04a3a8baf8be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 1, 49, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19a78bd2f5ed4703b78cc24909eec05b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e9a4e126c6ac141c1dc04a3a8baf8be
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_429da07986c980e850eefc0c9b62ae35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f72ee08bd49926f5a12dc9ea61f4d8a
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f4392f89b3e5025811646a8c01d1eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0c4179364846592b0ea1dbad6fb15cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d0cb74ea470c6e10a4fcbb8d24b0d7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f72ee08bd49926f5a12dc9ea61f4d8a
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df1be4735c4e443bea8a2b5130ab7cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.1887969821691513], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_e9c79e872c550070aef836f7f7d74595(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_65dee2a9c6c92594d123ba1448c73534(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9c79e872c550070aef836f7f7d74595
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.4264984428882599]], [[0.44378095865249634]]], dtype='float32').reshape([2, 1, 1]),
            ]


    class TestPrimitiveOp_0c8f9f329e4607fd5981954d1285cfd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c33a41d187e27ce6219ee00810edb64c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bca5aa3a098abebebfefe6a30f0aafe
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da901048bfcea6ff42543d252d3d0fdf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 1, 49, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_132b04cdb9a7db2907d4966964a2211e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da901048bfcea6ff42543d252d3d0fdf
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f3600a60b1c93de08bf1f64d55d0abb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_294dc6814d39e848a3237126858116fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7ddb30d5489fc8309baac56e82e6824(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_294dc6814d39e848a3237126858116fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71c9c8ce0aeb60341540af7c06fd6583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da3971fb2308b363a13d4c43202cde30(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 120, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e41a0b2e29b5c922679303917d8a4951(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da3971fb2308b363a13d4c43202cde30
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c4a1b8cf6f81511fc873b6c0551becb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5792c689efe13eb2b93b3c07c99f53d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d982d7c8daf6092692ed6a102cac48a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcbfa7479327897b5258bd3eb34e7449
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb539232ebc127cc65da7b92040cfdfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85c6d162df54e46e65ab0c79f3d347f7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eedf8855fb8a6d2917f6bc478d083426(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([2.2074105739593506, 2.0814871788024902, 2.0127875804901123, 1.9578313827514648, 2.0946848392486572, 2.2787415981292725, 2.061972141265869, 1.993729829788208, 1.838011384010315, 2.260199546813965, 2.0789196491241455, 2.2942206859588623, 2.0726144313812256, 1.9446386098861694, 2.2542762756347656, 2.0304219722747803, 2.349889039993286, 2.2931485176086426, 2.0853161811828613, 2.083437442779541], dtype='float32').reshape([20]),
                paddle.to_tensor([0.9763298034667969, 0.9317722320556641, 0.9069538116455078, 0.8245687484741211, 0.909284234046936, 0.5500822067260742, 0.9664220213890076, 0.7596622705459595, 0.7154883146286011, 0.8534694314002991, 0.6166922450065613, 0.596886396408081, 0.5243674516677856, 0.7685458660125732, 0.559390127658844, 0.5170217752456665, 0.612459659576416, 0.5005311965942383, 0.6033649444580078, 0.6565635800361633], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_ae4b4364a94b9fd736bbb16d0d420e9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([1.9475164413452148, 2.2945051193237305, 2.191253185272217, 2.280691146850586, 2.127012014389038, 2.0838818550109863, 2.1300153732299805, 2.147913932800293, 2.332383632659912, 1.930082082748413, 2.3858895301818848, 2.144876003265381, 2.0804991722106934, 2.0540261268615723, 2.0113155841827393, 2.045353651046753, 1.8870145082473755, 2.307952880859375, 2.000396251678467, 2.1889240741729736], dtype='float32').reshape([20]),
                paddle.to_tensor([0.023670200258493423, 0.06822778284549713, 0.09304618835449219, 0.1754312664270401, 0.09071575105190277, 0.4499177932739258, 0.03357797861099243, 0.24033771455287933, 0.2845117151737213, 0.14653056859970093, 0.3833077549934387, 0.40311363339424133, 0.47563251852989197, 0.23145411908626556, 0.440609872341156, 0.4829781949520111, 0.3875403106212616, 0.4994688332080841, 0.3966350555419922, 0.34343641996383667], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_05e191ff0bff976b3dac3a4832437c8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5503146648406982, 0.5240052342414856, 0.5073482990264893, 0.5036177635192871, 0.5244043469429016, 0.547767698764801, 0.5160642266273499, 0.5076965093612671, 0.49466651678085327, 0.5529568195343018, 0.5491458773612976, 0.5585044622421265, 0.5190911293029785, 0.49248918890953064, 0.5368063449859619, 0.5094084143638611, 0.5426266193389893, 0.5751357078552246, 0.5129085183143616, 0.5299163460731506], dtype='float32').reshape([20]),
                paddle.to_tensor([0.16715207695960999, 0.4314711391925812, 0.16430820524692535, 0.42761585116386414, 0.2671683132648468, 0.07692061364650726, 0.2710318863391876, 0.20835819840431213, 0.1385306417942047, 0.040651559829711914, 0.15243127942085266, 0.08873989433050156, 0.46281492710113525, 0.10008900612592697, 0.24598225951194763, 0.2274261713027954, 0.4416486322879791, 0.02435649186372757, 0.0623704232275486, 0.47828513383865356], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_56c44035b3130359867a2c57c573af4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7ef02dbd371b18049de283c3720d1698(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_271b3e4f3c01d5db82a146cc60057444(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ef02dbd371b18049de283c3720d1698
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2b09b9403d81ba9ac578985e9c16be7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cad6ad970e6dfaa6c460628936e57ed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118549b559d1818cf8b38191066565a2
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ff671850a6a2e71413497f44c4ba5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bf38c2d1d71b091cc340168ca57d463
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd0d33a51bbdc161eb7340edab8b1dfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f60684c8294f8ca2a10d5f53e87ff11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97f18b72e00452424b7f39a0e758f9bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58d05c01baf5469b79e4d75973043a16
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.9418007731437683]], [[0.9117066860198975]], [[0.9077053666114807]], [[0.9410808086395264]], [[0.90910804271698]], [[0.9616879820823669]], [[0.9368215203285217]], [[0.9499552249908447]], [[0.9581537246704102]], [[0.8683320879936218]], [[0.9321792125701904]], [[0.9070103764533997]], [[0.8749677538871765]], [[0.8305680155754089]], [[0.9019274711608887]], [[0.9057850241661072]], [[0.8383494019508362]], [[0.9794363379478455]], [[0.8647102117538452]], [[0.9678527116775513]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_dfdd9d5d4b6878ced3b2f402a933dccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0c4179364846592b0ea1dbad6fb15cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79bd0bb6ffe46ec93c2d68e992e20aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa295c28651efc9fdac0eebf2afbca6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c89639e795f07d6397c59bac331f1e5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_da265ffc8a481b0a48e0e6f4e191266d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0029604582a10483549d3f1c94229675
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb4a7587a0a3284f774fee668c0ccce3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_291e09dcea47470981b1cf5656085cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab519c6db51c79d95e9a5637cc4a9a1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ca84d26b459637197c1f6232a7e4396(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b30832961fb90a735538e57ea84da9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f75fc64fc005b783511c14608b37c46
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8da0793720b7bb979742620da14b3f62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c856c650e671fd48ff60cf0a187ddf1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19a78bd2f5ed4703b78cc24909eec05b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e9a4e126c6ac141c1dc04a3a8baf8be
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5157e556b6103dc99deaf682efcdf8b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5528deaaf12799258d18ea91a72319de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16fbfdcc49bf0f4b3864926bf5be4738
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a9dd0bfbaab298f183f8d7a28d5bb88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36cbb76860d7c2eaf20e8ddd45e0963c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6d5d0d376d4f1e2bd565bc95001e6c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe6a117b827b117a870672ba925176ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09bcddc38cdd7faa655524e2404ffd16
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_877866900b3920af257ec80e7673cf47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_118549b559d1818cf8b38191066565a2
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da4e977afc32eeb4a1db722b3eb2e50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a1cd420d046be81d9368d11da27a448
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f3005f95d9251700a8006871c8be27f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.009707748889923096], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9c87d1cc1c1d60edbb3c5a9fdb7ff8f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.24875834584236145], [-0.07860368490219116], [0.2262011468410492], [0.2045152485370636]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.003068208694458008], [0.1348257064819336], [-0.2989177405834198], [0.05067698657512665]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_44f9e0291852481c517c85c3c8fdbef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3149139881134033], [-0.12571868300437927], [0.0250491201877594], [-0.0416700541973114]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.1347169578075409], [0.020368844270706177], [0.3810873031616211], [-0.13437679409980774]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_c7167bab7670f480a767b177a407a0be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3149139881134033], [-0.04792898893356323], [0.2415425181388855], [0.2045152485370636]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.1347169578075409], [0.3491324782371521], [0.3810873031616211], [0.0530858188867569]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_dff53eec799090ca6ba7456afb606a31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8fa295c28651efc9fdac0eebf2afbca6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7e0dcce7f6803204f6cdcddc95fe1ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9270f7f853ee2669d437e7f25a0878c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bdaecb6291852062b2a484ce6881bc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_873a597832d3331358dc5785c3ab81b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d430f384fcfd2532ad572a291ef712b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_debcf1eb491afa9a54f9da48342acef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d430f384fcfd2532ad572a291ef712b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e87604173472cc6d864ca5a5b1ba8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe9a6aa03b64ce85ecc0cb515796f7d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3404b86090996e7cd122523633fc4f5
        def get_inputs(self):
            return [
                paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa64eb0655f9ff371c23755cc3b2949d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c7520a0add26c7ec0df49fd13cd683f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_160c397f0ce7ef0ebb1b017c4b3c7eb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f75038649ebd99879921c5587996ed7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0029604582a10483549d3f1c94229675
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2936b61b1a8a9b9f6ed10c7be150c0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2936b61b1a8a9b9f6ed10c7be150c0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2936b61b1a8a9b9f6ed10c7be150c0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2936b61b1a8a9b9f6ed10c7be150c0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2936b61b1a8a9b9f6ed10c7be150c0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d701756714324f91ef69b968cb5d3f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d701756714324f91ef69b968cb5d3f45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2936b61b1a8a9b9f6ed10c7be150c0bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fcea3c64150e4ebd99a0627e17ce190a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4550618d14b1bcae1287847ac2c62e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8da1da84cca5de716bb28720065f545a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c7520a0add26c7ec0df49fd13cd683f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_774f392570e38d7bb41289ceff311ea6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8133b005c68e26031574f74885ff1996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_285323f25c3c9da505864c8c3c0e55ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df6038c9a94a07acd90053988d474464
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f8d9da620c354a559958e60e8fc494a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33331224b71f35ab09d274602959b89e
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08c8475a1df4b49346fe647ac26dc9a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d91cb121a2ee85b3c13198abe48ff2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08c8475a1df4b49346fe647ac26dc9a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaa50d81fbe5a2f86f981141fee768d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8904fdb688cc4a664b2ea8afaa750a2a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09649503976106644], dtype='float32').reshape([1]),
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04b0ecb0ba9a8e883db4e80e16edcd51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06fe9d161f929580f6f7960ca98450bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_685055c975652e937f89e13739267da5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2, 1, 9, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03ec027d0ea008f2cf0b667e1455fec3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_685055c975652e937f89e13739267da5
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a135009773327994f28f56dd77795a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.1989731639623642], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2a135009773327994f28f56dd77795a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbfd641430878afaba6f85674075bd88
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.1989731639623642], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_34f06ec41b397503a89cd428e78e9a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d97c383c90a67344e00ce6c3ca26500
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75548ccbd4ad4d63ad4f3700b63b65e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0d8e766e44103241e3660819e7afd21
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.44187647104263306], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_684420a4d86ff4c4a742b62ef19d9418(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62266086d59a19263d5aa2a242ad94b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_263ecad1e31e10159db30f634e466841(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dc3a52051b0916b7ba8341b54baf69c
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fe1d4e2dc38043d06d4d2238597476c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d2b6b5e12caf0d5d6223db21e514af8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b56b6af65c31b1e021422b1e7649e66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d45bec6fe8ad7f77d6bb7542bc45c20
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f40376c6bff04045a48cb1174a80aed8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0053bb80bc501a5d5b2bfe7d3efd102b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e2a7e971c8415121d8e59282c480f9b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b13fc6f2a23ce193252a72dcd5289f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98bf7ee0754a8e0271aec4f409c2a142
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e18541c18515cf243d858ff7d7df5c65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d70dedbe5ad0044a31c9d5a64d40572b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dec94aad4ac782bbd51e9a2fa0946ec
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8f10757bb376a8a04966954ba5593885(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7973a804d5a20e9f3032c85567f34a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f10757bb376a8a04966954ba5593885
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dc369f4ba1bfe7d722a6659a709c4b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3267163f84119ff3d8b4c713b4030e43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d45bec6fe8ad7f77d6bb7542bc45c20
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c29b26bc894030c5d8452e22ce39f01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f899d3b1536ddb3c539f74900c470d1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0029604582a10483549d3f1c94229675
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7a39bd10c2d9357b001ef6484705169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_873a597832d3331358dc5785c3ab81b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21434a7cdb5255da06d1948e83a04f81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8133b005c68e26031574f74885ff1996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0384f88ae28b65fce8f10867fbc6ee6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c856c650e671fd48ff60cf0a187ddf1e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8987c121e146f72232ea66535011fd89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.026828795671463013], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2de205c4f6178cc0caec0f305894fc1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bfabe7a3c8f0e40f76c77d08e39adea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abe8ab9fb770121df500353cfb5de196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2093075066804886], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_15cfe64acc8adb2c476fd6aa9f1dc28d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0029604582a10483549d3f1c94229675
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7072e9b0c34093ad0c626b46da28bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93bcfd575c4928a37f2fd0a8a0d70f67
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6890b78be95ff68cf0bfd348da9440c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fc139bdd5b3d226683f1ba301088af5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_776c35c0ae1c669dece4c3d1ecdeec30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.34211570024490356], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_970eb97ae963c961831ac96a7b754add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_461078821b823568be9e3706294fa19f
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d444ee8be89111b74c51838f6f1b0477(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_890908ca6741743aadde4fb4791fc844(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1f191f8525aa1c6c00b82c7b170d94a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_890908ca6741743aadde4fb4791fc844
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4250d88c1c364110172232d6a406416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15c6f3d9f928aa2b9e67b70d9c07a248(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_395fd353e78cc3507bd39c26d608a21b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4c5ce0824c56b979083a22cbe6c83ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2979bb2303466178eb9c591ab29ffe42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6676265940b4d11cd20af766da8c0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_db4d424b0a1b21bbc91b0e68d72974ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6676265940b4d11cd20af766da8c0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_9c8889f9d60a8f83d3bb3f18995af003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6676265940b4d11cd20af766da8c0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_f249f76bb62a04005d0edad640cc9921(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d6676265940b4d11cd20af766da8c0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_8290c215a3a2cbdf43d3b95e9633900f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4490073621273041], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5e9f56d38ba719f188926a86ef66d23a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c7520a0add26c7ec0df49fd13cd683f
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e28365185a0a63f935d2b4b89a97f57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f5792c689efe13eb2b93b3c07c99f53d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a44b1b3c666f374b5c4b73304de5ab5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_396081d327ad7011cbde4dad47072200
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b476a8461b908e8664771da31a81a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c1a8b3906e4e4461ccd3599ef2fc833
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.24439510703086853], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_994b27b099bcc56939e63dc718803e7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ef83e0496dd2c02a6fc7411a136584a
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862885c90eb67fb21023e5538f4df46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862885c90eb67fb21023e5538f4df46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862885c90eb67fb21023e5538f4df46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862885c90eb67fb21023e5538f4df46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862885c90eb67fb21023e5538f4df46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fcf2361da35f618802f789f63fcc92a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fcf2361da35f618802f789f63fcc92a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94c75c085343e2de1204e347bb002976
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_862885c90eb67fb21023e5538f4df46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae6ad2a9660b58933a5dae0b7d8f5bd8
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8bc8bf55b2ad3018c0e3ccae2b7c3f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.11256954073905945], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_82202eecd121b2591e1c7ecf9a606e94(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 624, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e6c56d2d1084f23f641299cb6ba1abb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82202eecd121b2591e1c7ecf9a606e94
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2beee6623380a90ab9945b3b4a287d74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a1d990179688d9253d7d9521a5509f6
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f2b15131c728592cbc6d804f20edf05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43e766a887340c9d6b26f2dfd70837d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_132b04cdb9a7db2907d4966964a2211e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da901048bfcea6ff42543d252d3d0fdf
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a01d1b4fb6bbd7ba3f630f72a91a118(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa3902d40c4d8fa1d261fc4daf01e225
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_438a49d39931b5a78661044c2bd7c60a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20b634735eb46cd74bbdf81e3ecd192f
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.33400335907936096], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c4b568a66e2145cac52d2a01856d7880(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6b7732d26e82b6afe3f760a0e9a339c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6b7732d26e82b6afe3f760a0e9a339c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec2244486e0886bf5fe692135a5ec670(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1aafe3efd51a8608edef7251bb4f2b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b53fa98805c749c2f31a799815f2595(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 92, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bd991f069a1ab907179460ef02398d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4300bc4423ff2cedc6856f55e110b685(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd019831f471b4c7dbcfe8226da11d93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_0d567ba5d0e66b2ee462b475f0a44deb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8c73dd426c5a1a58192924a32944e53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b65ba8adb7f38526cfc12a698597fcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4dc76c124ac1aa57ef8117229920ec85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7848e4c2822bc4232ac24aa217004c33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e23f38c33e4202c93dc321647f27949a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12418050318956375], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_742d39c7e665a3a6e9c24c0368a1c9f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca1a04b8a2c408dd03fb16a5c7ec9065(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b4947366a510544811fe194efd0735b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_feca4a373793c241d71a23114303f354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.49104568362236023]], [[0.35629308223724365]], [[0.3221745789051056]], [[0.290326863527298]]], dtype='float32').reshape([4, 1, 1]),
            ]


    class TestPrimitiveOp_fa55e692b1197bfd54d046f655e9f059(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cce6021f9b6567f1fe682e874157c67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_479de46fc86cacc3d09389795a955fa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_231a795a24b8177947ba7979a7ea3202(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3156996f01e1b04532cb6d0088c09d0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0cdbb480e49cc83875221767a0e98cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1e65a19b51b3a38afa3dccc9063be10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2802686095237732], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b124d8fc95c9e6fc2b94dc20e0e300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.9013332724571228]], [[0.8568813800811768]], [[0.8735890984535217]], [[0.9094127416610718]], [[0.7574430108070374]], [[0.9139612913131714]], [[0.9421039819717407]], [[0.8919312357902527]], [[0.87510746717453]], [[0.8356452584266663]], [[0.9212054014205933]], [[0.8290369510650635]], [[0.8940802812576294]], [[0.7877908945083618]], [[0.935991644859314]], [[0.8388787508010864]], [[0.9542338848114014]], [[0.889322817325592]], [[0.8550490736961365]], [[0.8559415340423584]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_7b4b98da6beb069b3550e1910548f4ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e00a7d4bbecd5c8fd2f4183e2fbd328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_874902e8ae1ac3973c6bd085c477c755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c33e5438b35c66af5dbc640161cd826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_080d7e236061cd000e50cc90ac6ee7d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_080d7e236061cd000e50cc90ac6ee7d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0007197b604f18d29125e524034c4f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24372179806232452]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_86d2054aa3b92624178c32b238cf9d1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25b3df4e0ed033ca76a2bdb49556eb71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.06334684044122696], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_09d030a477e87212872e549e516580ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e65f90cea937383335fc1882a94c8c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a890da18f1285e142d4fb554379f7db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3e88d201acae2b4ef86aa9da8b25bfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75b8b4a59de1678d4ce248e57c6ee172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ff11df2d80ea8dc30f22d3c8c3f0215(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b13b28adde94467e0b9b99a9a1f7808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc380299e2bbfef843493098f1a16cd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c01e7865e9224ff4a8b737109a8411b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e82a60b3f44457afe272790ba7378d17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3951542953fd54cf33e334f4a2dbc883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b72e6b0bf99a853d0e0fd50ab495b33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2ee997bb79fa624b27b423a302feda1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.221171572804451], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_83de3c5a8b38b031561051e051cae433(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e89aae86b91876d7ee6301d9dc3ffde2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb461c5b79831508f5031bc2257c8af5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c83adcd7c8d07451e3952774db8d82b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_192399e3719e0f21f194027a7709e090(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4209981858730316], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_802cdb59e9e013d11580ca3cbc18a4ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea883af0807c40560f552dc560970e3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.6297255158424377]], [[0.9214640259742737]], [[0.8490052819252014]], [[0.7044241428375244]], [[0.825300931930542]], [[0.770827054977417]], [[0.6619652509689331]], [[0.7667979598045349]], [[0.7892367839813232]], [[0.9679443836212158]], [[0.819905161857605]], [[0.6637662053108215]], [[0.921315610408783]], [[0.720565676689148]], [[0.8479679226875305]], [[0.728390634059906]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_dd7f181b1ce63ff4826391dd0723be99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b284895e75d565b1986187725f8b982d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3479c73787fec5e1be7dd664a2920afc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f955b035f8752d938ad6995f3a2f7d7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71ac099fe8b14c90f0a00a455feaedac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.12735623121261597, 0.1039394810795784]], [[-0.010912656784057617, 0.05330643057823181]], [[-0.048233628273010254, 0.2193552404642105]], [[0.2782248854637146, -0.4064857065677643]], [[-0.1268930435180664, 0.32784441113471985]], [[0.19737398624420166, 0.012657210230827332]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_ecedf5ec28eb8cffd970c6f31df7563e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[-0.06997059285640717, 0.07948741316795349]], [[-0.27302616834640503, 0.08186653256416321]], [[0.12233045697212219, 0.23984484374523163]], [[-0.05789706110954285, -0.15295448899269104]], [[0.11353802680969238, 0.14615774154663086]], [[0.1804485023021698, 0.03878364711999893]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_e7917e9a00157f4d082a9e82af1a60c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.12735623121261597, 0.1039394810795784]], [[-0.010912656784057617, 0.05330643057823181]], [[-0.048233628273010254, 0.2193552404642105]], [[0.2782248854637146, -0.4064857065677643]], [[-0.1268930435180664, 0.32784441113471985]], [[0.19737398624420166, 0.012657210230827332]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.12735623121261597, 0.1039394810795784]], [[-0.010912656784057617, 0.05330643057823181]], [[-0.048233628273010254, 0.2193552404642105]], [[0.2782248854637146, -0.4064857065677643]], [[-0.1268930435180664, 0.32784441113471985]], [[0.19737398624420166, 0.012657210230827332]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_6f4f98b3264a451a39d41af1110cc320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[-0.06997059285640717, 0.07948741316795349]], [[-0.27302616834640503, 0.08186653256416321]], [[0.12233045697212219, 0.23984484374523163]], [[-0.05789706110954285, -0.15295448899269104]], [[0.11353802680969238, 0.14615774154663086]], [[0.1804485023021698, 0.03878364711999893]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[-0.06997059285640717, 0.07948741316795349]], [[-0.27302616834640503, 0.08186653256416321]], [[0.12233045697212219, 0.23984484374523163]], [[-0.05789706110954285, -0.15295448899269104]], [[0.11353802680969238, 0.14615774154663086]], [[0.1804485023021698, 0.03878364711999893]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_6e59b7a45f3fd2bcd3e776dff65c8251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.004442228935658932], [0.0001610954204807058], [0.011329323053359985], [0.1195206269621849], [0.04344525560736656], [0.007736477535218]]], dtype='float32').reshape([1, 6, 1]),
                paddle.to_tensor([[[0.02352176047861576], [0.1643000692129135], [0.1283266544342041], [0.1822909265756607], [0.12502682209014893], [0.02151423506438732]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_fc98ef7372bab942e450e25370328635(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.0011875410564243793], [0.02315785363316536], [0.01951729878783226], [0.004374376963824034], [0.006339387036859989], [0.0062875086441636086]]], dtype='float32').reshape([1, 6, 1]),
                paddle.to_tensor([[[0.02352176047861576], [0.1643000692129135], [0.1283266544342041], [0.1822909265756607], [0.12502682209014893], [0.02151423506438732]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_148d8c420c8c9c43d8e54f08503c8a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e8baee1347fa82de1e363821f70ed3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06d777b49e3415ef43eac3d6a1ef7f15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b372733ab90a4a815cf7ca6464aaecef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90cb455d2282d02a6d0c5ce4395633ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b7dc0e180e99455e5e6b23913262cf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_effc7d326ab7105447ce0dd4feb20226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b62209bad0500a21151aed10f5c61bef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9fc1fa00a98dc844fb298d4eb9ed3a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_263d7a34ca1065aa378778e3fdabf4c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([1.8156734704971313, 2.0851945877075195, 2.0087790489196777, 2.175837755203247, 2.072892189025879, 2.226891279220581, 2.2403757572174072, 2.028991222381592, 1.9627472162246704, 2.1859636306762695, 2.071002960205078, 2.0156030654907227, 2.1725573539733887, 1.8901900053024292, 2.30122447013855, 2.2802863121032715], dtype='float32').reshape([16]),
                paddle.to_tensor([0.6253926753997803, 0.5155894160270691, 0.8181114196777344, 0.5579231977462769, 0.9746584892272949, 0.6788683533668518, 0.9904800653457642, 0.5969765186309814, 0.7701771259307861, 0.8957492113113403, 0.6820752024650574, 0.5907605886459351, 0.7641318440437317, 0.8520417213439941, 0.9319855570793152, 0.5559881329536438], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_42752d9423807dcd83e1fdfcd2c8ade4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([1.8358906507492065, 2.0165863037109375, 1.986639380455017, 2.0012130737304688, 2.1091840267181396, 2.1463732719421387, 1.894654393196106, 1.9875061511993408, 2.0823562145233154, 2.1248931884765625, 2.0183653831481934, 2.0908870697021484, 2.0069997310638428, 1.9543657302856445, 1.935641884803772, 2.2911624908447266], dtype='float32').reshape([16]),
                paddle.to_tensor([0.37460729479789734, 0.4844105839729309, 0.181888610124588, 0.44207683205604553, 0.02534153312444687, 0.3211316466331482, 0.009519957937300205, 0.40302348136901855, 0.22982287406921387, 0.10425077378749847, 0.3179247975349426, 0.40923941135406494, 0.2358681559562683, 0.14795830845832825, 0.06801445782184601, 0.4440118670463562], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_b1e51bcc452a2eb745c327afbd5f4ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.4558117389678955, 0.5129899978637695, 0.5011880397796631, 0.5246601104736328, 0.5184529423713684, 0.5502585768699646, 0.5592711567878723, 0.5030679106712341, 0.4975590109825134, 0.5448992252349854, 0.5135670304298401, 0.511603057384491, 0.5333768725395203, 0.4749213457107544, 0.5690898895263672, 0.5712788701057434], dtype='float32').reshape([16]),
                paddle.to_tensor([0.42562201619148254, 0.3016831576824188, 0.2649511396884918, 0.48494216799736023, 0.07584431022405624, 0.2885129749774933, 0.08532002568244934, 0.48674431443214417, 0.20337152481079102, 0.09332582354545593, 0.03715044632554054, 0.47536739706993103, 0.3538185656070709, 0.22236795723438263, 0.07157617062330246, 0.08841179311275482], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_3673a6d8f68c006e0d8f91153ac285c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_048f4e6dcb867f3e2c222d6567714d41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.38461145758628845], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2306459381e590223caa0e743bd18be7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7800b55f6c353993f632a038639d1a4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.09994722902774811], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_94a1815986081e997a97637b30a0cd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.044002752751111984, 0.4828374981880188, 0.2906877398490906, 0.013120715506374836]]], dtype='float32').reshape([1, 1, 4]),
            ]


    class TestPrimitiveOp_63f38df7c27d0c6c3d9a7506d3c507d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b701aabf14a58b8aacccae965667aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3522210717201233], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1146e61e3071585da61715e7b9a1d501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.477573424577713], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ff6e2bc92f23adad09009e3e61cbb22c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.19731079041957855], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd5c5f6fd89e90f10bb9f0afd3a0cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47f7f58552809fb3e10ff0e6eaed0e1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8725aa8a32d2bde2e3f3a23260cd7d4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.21855682134628296]], [[0.10699958354234695]], [[0.44307953119277954]]], dtype='float32').reshape([3, 1, 1]),
            ]


    
    class PrimitiveOp_5d2532e967b23c0ebfaf74bcbec2b27c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_563b489318049ccd3e70f4d6af744f87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d2532e967b23c0ebfaf74bcbec2b27c
        def get_inputs(self):
            return [
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a201b12b97258b113ab17ad5195a1fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b7a4a7f1727a8958e0762cd885b120f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b65ba8adb7f38526cfc12a698597fcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7d89f6504c544bf6bba7bbd50077c58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.04578159749507904], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_394894950dcbdea06001d5fbdcb86557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3acdf326cd11b6ef38855cf30f8fa6a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34552656d027bf9333b897fa317c816d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b3f7a29d8dda21babe6b7512ac1c7a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ecfd3a0adc0609c2f664a87f8dbe1f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bb8543be35a918df003928a6f6f543c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_977a33d8d59416c171532353eccb992a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_537117464c98516e76debf103d25d24d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3992e77119167fefdaaba334f629a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3992e77119167fefdaaba334f629a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3992e77119167fefdaaba334f629a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3992e77119167fefdaaba334f629a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3992e77119167fefdaaba334f629a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eac3b63301944e970382c4423f7899ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eac3b63301944e970382c4423f7899ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3992e77119167fefdaaba334f629a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acbc05f57052ed7e3969284d6ba704a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c0822d7672b3bdaad368e2253ea0466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_111c3fe767d6ffbc7909d773da5b264d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 6, 9], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.15479785203933716], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_35200d81b3d4f0d05210b58ee3724e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_800063e2d6fe6c42a439a15592ffa615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_19285ac8547adbb033173825d51ce143(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db4cc4d23b5b14284fbd412104e3e0ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 5, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3938605785369873], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_59ddf757920929c8f5643a163ad88a5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 1, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4936404db4557ececda8d277c1d2a4b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88e2b620090c939673479d5ad1dea751(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74a42bd83390064f5c59f4eadd66be96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e02b0c28477f0ffdfebec777210793e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93ab1c350e23baf90c47ae068fe04b7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4005f3c369ede421b4fb7ff247891d27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9240ab2397244d23bb6cd2a8be6bb5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24fb2104dbd0635b9f4051e7c7bae641(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9c2e4c366a8fa13577b709dfb5ee062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04b66f730b3d800f159880fa25b04d0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccc7b537e3d9756f22aeaf6b7b228d20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6341d2f169b55d1f43855f78f7f839c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.21154245734214783], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.25445711612701416], [0.21916604042053223], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_8741fb547475a6408512aa977372d495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.009391963481903076], [0.19519536197185516], [0.21154245734214783], [0.08582274615764618], [0.24007847905158997], [0.2587222456932068], [-0.2994767725467682], [0.45847052335739136], [0.17372050881385803]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.02195553481578827], [0.15992991626262665], [-0.21389555931091309], [0.14502179622650146], [-0.008672813884913921], [0.25445711612701416], [0.2555086314678192], [0.13578519225120544], [0.3360256254673004]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_169b4ec67da56b44d3c0ae062392507f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.16356325149536133], [-0.1396092176437378], [0.4566301703453064], [-0.2960786819458008], [-0.07350330054759979], [-0.09332668781280518], [-0.0065939947962760925], [-0.2193688452243805], [-0.13021743297576904]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[-0.04090815782546997], [-0.14576321840286255], [-0.1428469866514206], [-0.11748522520065308], [0.3009539842605591], [0.2908601760864258], [0.3797423243522644], [-0.4335433840751648], [-0.18733468651771545]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_3c56ddbfb092e15f94e018b5b60c9806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08131581544876099], [0.19519536197185516], [0.4566301703453064], [0.08582274615764618], [0.24007847905158997], [0.2587222456932068], [0.07948271930217743], [0.45847052335739136], [0.28396111726760864]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.31203949451446533], [0.15992991626262665], [-0.09310540556907654], [0.25870460271835327], [0.46219155192375183], [0.2908601760864258], [0.4160849153995514], [0.13578519225120544], [0.3360256254673004]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_6ab841715e0ea2beb2feec6ea91be8e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8346992402449a471ef8b1404220a73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ac3071743993b80f62147fdaaefa251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62c4aa44126a34ad4951bc015d9a76c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33dc7374ec873fb06bff220fef2b72ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4b49591d65706836e5c3dda9f98a4b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4731403589248657], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f4b49591d65706836e5c3dda9f98a4b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4731403589248657], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0bced0c6f97c8e7739a380135ebf6063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b141eae9c16e4a8348c5afc2d54992f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.37376904487609863], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5bae75c67b10563637364d24bad6ca67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85d08488a4c8ff81704d43fb1e3c700c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aac41b4d6c9800a50775806269633b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a543ca4c8e03c9ab13646a6341e006db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a543ca4c8e03c9ab13646a6341e006db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a543ca4c8e03c9ab13646a6341e006db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a543ca4c8e03c9ab13646a6341e006db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a543ca4c8e03c9ab13646a6341e006db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78c20f0583fad95ad05d67e4c85fb6f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78c20f0583fad95ad05d67e4c85fb6f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a543ca4c8e03c9ab13646a6341e006db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00a25c7062298fe912cd839327376bb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d434c75b2944d768c82dd8156999427(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3c431f2e35000a7b3a88edb1d5348ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a16bff1fbe607c714b72b5634de63526(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a16bff1fbe607c714b72b5634de63526(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a16bff1fbe607c714b72b5634de63526(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f5ad6d4e1ebec486da9726cc55b3838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c36af01679a0f98a596c5eccde0d9142(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.33328139781951904], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_04e58ee6a8c86d2d5cefd6ac7831f895(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56f8abb87fdc9ac77c5f9a42ef664ac8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dafce0fb35143467b327d53eeb8ea2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c67245989d2701f8e6e72d12cedd1c93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_752a86769e1838e98359df53b7c9bc40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd7f181b1ce63ff4826391dd0723be99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f2e60e3a3c00db7d8004353f81257d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_835a1eb1c6c5be9b799d67abf18fa574(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34a3cc0bc6d6b2bbe0fd3cb0f4443db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c96a223e319353078fbd42c91f55965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4dafed9690a5487ff874f24ef2d57a66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77c509c2ff32a34668aadcd5031aa471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 10, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.31139373779296875], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c817d75b987e115b0e119b3cfdd0e65c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02f9a4bafd0dfa77aa959b55560123ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.8605836033821106], dtype='float32').reshape([1]),
                paddle.to_tensor([0.3450847864151001], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2a06b6f7a7a6ba03208c30c9634d8b86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6908892393112183], dtype='float32').reshape([1]),
                paddle.to_tensor([0.02924232929944992], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3c96d0a854754e3e96cd56a830daf4fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_733b0c227453319d928c88572f456177(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc4d6f6d852e0a9729edd5248e83e90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ae8ab248dd3c2be3616d93069237334(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d343a1d326f0147a76ff088201dc163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b60c2f943bf985065b73962abe9d7bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0659c4400b8cf1f7af657a109ab92c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_108fb910a1025daaf3f585d66b96e265(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12313771992921829], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dc380299e2bbfef843493098f1a16cd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_faa8abe30f3b1baac1330f53b696f883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 12, 18], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12150554358959198], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_cb72061f771b73dd6aa689ba11e9b0d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccc7b537e3d9756f22aeaf6b7b228d20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d36fa7ecea096384d47700594f67ea7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a79541d4510070b5dc511dc585a5bd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10905a55e0a8ce1415fc6a8332cb9596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2766314744949341, -0.07481521368026733, -0.03476780652999878, -0.2757680416107178, 0.0, 0.02061089128255844], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, 0.1621771603822708, -0.20417125523090363, -0.3648090064525604, -0.0037841498851776123, -0.3224535882472992], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_50fbb131c90b9b45ff3d00dae13da0a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, -0.012133318930864334, 0.007098586764186621, 0.1006026640534401, -0.0, -0.006646055728197098], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_11a40ea4ae7719990bd6ab8b4f233154(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0, -0.0, 0.0, 0.0, -0.0, -0.006646055728197098], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_bae564c12bcbfb2080ca48796a874d99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.21817441284656525, 0.0, 0.0, 0.05182693898677826, 0.0, 0.02061089128255844], dtype='float32').reshape([6]),
                paddle.to_tensor([0.0, 0.42682701349258423, 0.020339012145996094, 0.0, 0.0, 0.02947470359504223], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_3473af602531c81d140c934efdeb3e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2766314744949341, 0.10538440942764282, 0.14060714840888977, -0.007649481296539307, 0.0995698869228363, 0.10518306493759155], dtype='float32').reshape([6]),
                paddle.to_tensor([0.09058192372322083, 0.1621771603822708, 0.031248152256011963, 0.106215700507164, 0.17448961734771729, 0.0300922691822052], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_741480208c0362158e60cea4e623c850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.0168093740940094, -0.15478301048278809, -0.01066252589225769, -0.29785677790641785, -0.09646612405776978, 0.032794758677482605], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0168093740940094, -0.15478301048278809, -0.01066252589225769, -0.29785677790641785, -0.09646612405776978, 0.032794758677482605], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_efe8c5765fa30340be76cf124e5574e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.053348422050476074, 0.0016325414180755615, -0.22996483743190765, 0.2222844958305359, 0.0067261457443237305, -0.35223710536956787], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.053348422050476074, 0.0016325414180755615, -0.22996483743190765, 0.2222844958305359, 0.0067261457443237305, -0.35223710536956787], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c640aa10d6149834fa01563ad2792fd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.21817441284656525, 0.18019962310791016, 0.17537495493888855, 0.3199455142021179, 0.0995698869228363, 0.10518306493759155], dtype='float32').reshape([6]),
                paddle.to_tensor([0.21817441284656525, 0.18019962310791016, 0.17537495493888855, 0.3199455142021179, 0.0995698869228363, 0.10518306493759155], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0e989e72105bbcca227c75e6d34aef52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09058192372322083, 0.42682701349258423, 0.2557584047317505, 0.47102469205856323, 0.1782737672328949, 0.3820205628871918], dtype='float32').reshape([6]),
                paddle.to_tensor([0.09058192372322083, 0.42682701349258423, 0.2557584047317505, 0.47102469205856323, 0.1782737672328949, 0.3820205628871918], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_52ea9610077e311ad3d94a938bbf41df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0340176522731781, 0.28506091237068176, 1.14909029006958, 0.024235691875219345, -0.05860188230872154, 0.2763666808605194], dtype='float32').reshape([6]),
                paddle.to_tensor([0.08393514156341553, 0.7033591270446777, 2.8352646827697754, 0.059799134731292725, -0.1445942521095276, 0.6819069981575012], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_8b46e7b2839d0c4ab43bd38b6f520c68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0028471469413489103, 0.16701386868953705, 0.7651466131210327, 0.001447176095098257, 0.008402298204600811, 0.15857239067554474], dtype='float32').reshape([6]),
                paddle.to_tensor([0.002855276456102729, 0.20050019025802612, 3.2579751014709473, 0.0014492734335362911, 0.00847349502146244, 0.18845637142658234], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_7f4228d15b903cb28e80004e16325f7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c98fae151bd450c3145afafdc4fefc7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_288f0da40dec3aec0c8ce6f520d5e1a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f71edbd0d4c6e28ca62a2e28a8001475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f71edbd0d4c6e28ca62a2e28a8001475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f71edbd0d4c6e28ca62a2e28a8001475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f71edbd0d4c6e28ca62a2e28a8001475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f71edbd0d4c6e28ca62a2e28a8001475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59b131e93d6ced5b74143fe3867d81ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59b131e93d6ced5b74143fe3867d81ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f71edbd0d4c6e28ca62a2e28a8001475(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aac41b4d6c9800a50775806269633b58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8573ea638f1783e612e596e11603eef8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2fb55cc04ed8b18630e5d2f054e502b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_227582c34c15df623c514d8b58d9713f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acbc05f57052ed7e3969284d6ba704a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54ae71bf3d96f1a3946996e8a1b5826b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c39cc299177b789e3d27d1a510acfe4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ccf9915ff9f4df64d5e923d3e46a63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4205133616924286], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c7ae8ac84a294bd7d8192ca6aa89c4d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8599860040d72d7f74c4d25cf6f913d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46ba60237f4f37a6a3f0d0a0dcdb5ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f38df7c27d0c6c3d9a7506d3c507d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2393d59d1d65e9786a9ae92cd6e228ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b97c75aadf8e0468f15102a5971ba17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_faa689f9497b967e0f4f3660c30fe6fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc09087a6676ae3cc47fa41a2af66884(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b9f231623f6a5be2e26931120ad2959(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0641dfcaf9d8ab2c33aadcc1e7fb3fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 13, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.30836474895477295], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ade5edae99b9ddceaffd249ff1f4292b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e60e9b8fddce4848be7b48b2cead759(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([2.0022754669189453, 2.240567207336426, 2.111360788345337, 2.1870291233062744, 1.9671711921691895, 2.0732555389404297, 1.9710893630981445, 2.024165153503418, 2.334258794784546, 2.063455581665039, 1.9864293336868286, 2.0223824977874756, 2.1271204948425293, 2.148427963256836, 1.9668179750442505, 1.997074842453003, 2.1886327266693115, 2.188886880874634, 2.300858497619629, 1.8540812730789185, 1.9147701263427734, 1.975813627243042, 2.1764822006225586, 1.938555359840393], dtype='float32').reshape([24]),
                paddle.to_tensor([0.8168052434921265, 0.7831318378448486, 0.5267834663391113, 0.96436607837677, 0.7379273176193237, 0.869096040725708, 0.5500746965408325, 0.9029768705368042, 0.5807815194129944, 0.6169620752334595, 0.5883928537368774, 0.7845883965492249, 0.8679076433181763, 0.5795742273330688, 0.5975856184959412, 0.8808311820030212, 0.8277290463447571, 0.6553834676742554, 0.5039186477661133, 0.8484286069869995, 0.8710693120956421, 0.6950018405914307, 0.995990514755249, 0.5658894181251526], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_309c333c6cb3f4fe8817a77f77429087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([2.219653606414795, 1.9027743339538574, 1.973353624343872, 2.1888537406921387, 2.2433595657348633, 2.0183637142181396, 1.9209812879562378, 1.9374973773956299, 2.027986764907837, 2.10695219039917, 2.2992305755615234, 2.2776315212249756, 2.1760663986206055, 1.852568507194519, 2.2449796199798584, 2.340653657913208, 2.0092720985412598, 2.016832113265991, 2.133084535598755, 2.299717903137207, 2.0900368690490723, 2.204993724822998, 1.8239502906799316, 1.931168556213379], dtype='float32').reshape([24]),
                paddle.to_tensor([0.18319472670555115, 0.21686814725399017, 0.4732165038585663, 0.03563392162322998, 0.26207268238067627, 0.130903959274292, 0.44992533326148987, 0.0970231369137764, 0.4192184805870056, 0.3830379247665405, 0.41160717606544495, 0.21541158854961395, 0.13209235668182373, 0.42042574286460876, 0.40241438150405884, 0.11916881799697876, 0.17227095365524292, 0.34461653232574463, 0.4960813820362091, 0.1515713632106781, 0.1289307177066803, 0.3049981892108917, 0.004009498283267021, 0.4341105818748474], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_89b8dd68e2fa7316eac740f19805f0c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5105245113372803, 0.5418276786804199, 0.511513352394104, 0.5467735528945923, 0.5098881721496582, 0.5165175199508667, 0.4871361255645752, 0.5039390921592712, 0.5514659881591797, 0.5200290679931641, 0.5287951231002808, 0.5193415284156799, 0.5333964824676514, 0.5060102343559265, 0.5196885466575623, 0.5095046758651733, 0.5394335389137268, 0.5323984622955322, 0.5544072389602661, 0.4804067611694336, 0.4843418598175049, 0.5114282965660095, 0.5437671542167664, 0.48383718729019165], dtype='float32').reshape([24]),
                paddle.to_tensor([0.43111735582351685, 0.2893909513950348, 0.15826816856861115, 0.4119057357311249, 0.24751590192317963, 0.3494945168495178, 0.3399348258972168, 0.38630738854408264, 0.3234490156173706, 0.09757565706968307, 0.2129882127046585, 0.46143293380737305, 0.20221589505672455, 0.08494094014167786, 0.43189936876296997, 0.3254511058330536, 0.13017435371875763, 0.2705845236778259, 0.48569831252098083, 0.06358955800533295, 0.33293449878692627, 0.40925174951553345, 0.0861608162522316, 0.37648701667785645], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_953be0c927ff899b9099194618e224c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f71ea6adb430c6772ff31d1981c6eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb452955bd2f6043110982191e8964f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6cff4b1253e8cc931063da04499ba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6cff4b1253e8cc931063da04499ba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6cff4b1253e8cc931063da04499ba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6cff4b1253e8cc931063da04499ba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6cff4b1253e8cc931063da04499ba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f37afc99d385f234bc38b321b91bbf3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f37afc99d385f234bc38b321b91bbf3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6cff4b1253e8cc931063da04499ba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_670284e0697f88728dc5e83d341bfc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_670284e0697f88728dc5e83d341bfc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f130a01009b3ed73dbe56ed6f7138712(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24634625017642975], [0.2473631501197815]]], dtype='float32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_1337383dacf41e35a649a7ac25ab58c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_caeb6e78fef3626e68a15bfe88f85ade(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fccf8ef7ac2ccd2a025c930acd45a01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24e4dff6281da5d1bbf108b591523c71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1cd708f5ef4df3a184f694d4dbfd4b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_148d8c420c8c9c43d8e54f08503c8a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_382908dd08f6d75438844a1a24ddad92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_163e87b2d6e95d53191635b8612414bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.12146636843681335], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0905c3585914dfcb48f74e2a637af6ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cb656613ac8225fd58de9cdf89c6d90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e930ca8185afe2dea3f9a8781fdcd9aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([2.0273215770721436, 2.051156520843506, 1.8777961730957031, 2.1210427284240723], dtype='float32').reshape([4]),
                paddle.to_tensor([0.9491412043571472, 0.7706915140151978, 0.8685111999511719, 0.8350591659545898], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_ace8d5003d862f0bd176a7dea39a34f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([1.9933178424835205, 2.198335647583008, 2.116931438446045, 2.207944393157959], dtype='float32').reshape([4]),
                paddle.to_tensor([0.05085877701640129, 0.22930851578712463, 0.13148881494998932, 0.16494080424308777], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_2998c4efe4c71424b836428fcef81d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5063980221748352, 0.5212265253067017, 0.4773099422454834, 0.5338440537452698], dtype='float32').reshape([4]),
                paddle.to_tensor([0.493753045797348, 0.14767983555793762, 0.4444725215435028, 0.12316848337650299], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d96aedbd867686054c0f6fd8caebaee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f272766736e95e13409be386edda3871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f8efe443399e07f5064c63b180eca8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff51deb57e63bcace2ae85e85d01185a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecf0fe0fe2feda6c25b6c47c56b2c67a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a79541d4510070b5dc511dc585a5bd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38ab66cb7d8bdf9c7b86b98e2aab5bef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_083b0abe5262df9e8fe1b52981200950(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 * input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c8ee353d9d224148c2e7e4daba9b14b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_083b0abe5262df9e8fe1b52981200950
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2909002900123596], dtype='float32').reshape([1]),
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce21b5a12358cb117bdf05ac28ba418a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07224521040916443]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_be91cb14cbf4db5aaee9cf1c76ffcf72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.09675587713718414]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.04453441500663757]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_260772738db026605ac0d6fa50aa3687(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10698825120925903]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.14122912287712097]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_6af37a2b9ab34ce8f5429412a4d9c113(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13149891793727875]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.19459663331508636]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_caeb6e78fef3626e68a15bfe88f85ade(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4cbbc94be80c5c26af3114cc037bd02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.11981570720672607], [0.14607250690460205], [0.1565159410238266], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d23e1e6d4b915830484f5d4b6792e6f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08189225196838379], [-0.1403680443763733], [0.07632625102996826], [0.03819593787193298], [-0.01861479878425598], [-0.09420964121818542]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.1627451777458191], [0.25780361890792847], [0.2799569368362427], [0.068025141954422], [0.0581510066986084], [0.08422863483428955]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d57996aab73a74e636b5cdf391eee717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.1917412281036377], [0.039921388030052185], [0.13284483551979065], [-0.3065911531448364], [-0.05574962496757507], [0.30769357085227966]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.1562783122062683], [0.20795682072639465], [0.29048770666122437], [-0.0940697193145752], [-0.0030398517847061157], [-0.08507192134857178]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_9868fbf45f195a32b3e16aef8564d185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01215440034866333], [0.039921388030052185], [0.3015764653682709], [0.0914648175239563], [0.2659372091293335], [0.30769357085227966]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.19920778274536133], [0.31968793272972107], [0.41392871737480164], [0.08479256927967072], [0.21667928993701935], [0.09567250311374664]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_a951722da0ac24afe5a73a9d8fd27d5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67d81d3291981af9d3dc50e35f05f734(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8200b69d2e4334a5efb95c7baa41eb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47f0fd4013b47c3fd0050cfc1af3f5eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_4e5e3622dc4b41a4446a8b236dd6d508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_510223e1ee74cbf5b8f9c0666d0d128d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3b553626e8596d3a491ac40cfd8c5eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.7649943828582764]], [[0.7536299228668213]], [[0.8622885942459106]], [[0.8777257204055786]], [[0.7248691916465759]], [[0.7289198637008667]], [[0.8996758460998535]], [[0.8994820713996887]], [[0.9005266427993774]], [[0.8886114358901978]], [[0.8354483246803284]], [[0.8511449694633484]], [[0.8733361959457397]], [[0.9242442846298218]], [[0.8901932239532471]], [[0.7633470296859741]]]], dtype='float32').reshape([1, 16, 1, 1]),
            ]


    class TestPrimitiveOp_58fad0434382415dd0a83e02fbb55af6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02a7e0c40263e67fdea1a2af3793933c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_396fc1c30766042bfb77b4fc3b99d4a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_134b956b40ee3b2ad9d4a5069f874859(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 60, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b817ad105e8f41c2f40515329c43736b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 10, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.34778791666030884], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a7170632bc02ccaae66d4106b1facf56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec713e4c96d354cfa93dd9e8ed1daa04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8514b99b9f1fe5a4efcaa9a34833bcd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8514b99b9f1fe5a4efcaa9a34833bcd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a506a29bf877e566143fadccbe38dc36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.24644309282302856]]], dtype='float32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_1e2c7e83a1f0efc4507bde41c5a2f6ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50f6fee1aa0230d4ff566820f5d7969a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cb9ea5944484f9f713389a60ab6b4f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f52841e0c7dc309bfca50c8d283b547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc5f75ce8b6646aeb8e7f86217a99fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6011af65655d76109e6bb04aae71251e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bdd75551585415732d5a1ed2d65098e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa55e692b1197bfd54d046f655e9f059(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_511df2575a6a6fc682472e574ff70887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf77d44f323ef38e42675394064a1de5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_017ec8d2ca5b881bba61a00ca002f5ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe3c3992fa794182acc9a55added4a1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd533d062a6c48df8bc479301c401626(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cee5eebb5c232dbeb5d89c96a3c937a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0a556135aff7077925a3d14bfee23ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9672295d91743b005bf86e524f041d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7602f0f687fbf35cda11932c70b9466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ade5edae99b9ddceaffd249ff1f4292b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60025833755fbdd51d4eb42899d9b368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf579ce78f05a0715e1bfcd40616d1b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abaffcab47ce3e0f6544f788af64debf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22d3c322d1ff76411584d1c9cc9f7dfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ba0db1a2cd543ce21cb3ebf7460ca4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 44, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebaf13317b84b1e425da0f1763bfe199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3055a7d4ef49da78c2e88331319a960(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_738a4b36459639fecc421392caae3813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.40281563997268677], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c27bd5175c8678d603d94a4b661e382d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92d1865c6c3fe7c70a22221e8e302122(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5adc89ebbee6c02888c17fff01e442af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_693ed7ae37911d435b060aadd39b0ea8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fc7c9dcbc4acb57489a46f865510b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fc7c9dcbc4acb57489a46f865510b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fc7c9dcbc4acb57489a46f865510b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fc7c9dcbc4acb57489a46f865510b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fc7c9dcbc4acb57489a46f865510b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a951035aa211bef4812f3d776325026a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a951035aa211bef4812f3d776325026a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fc7c9dcbc4acb57489a46f865510b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fbcaaaaf45daff8beafc3ecc6639575(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0198c549b8ce297c7e4eb6f3b05331f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.9709991812705994], dtype='float32').reshape([1]),
                paddle.to_tensor([0.0655159056186676], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fe94b06811c1f45347fa06dcedc5b41b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.9603669047355652], dtype='float32').reshape([1]),
                paddle.to_tensor([0.24741116166114807], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b6b53edcb485edc44aa0f5baf15c826e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6136724352836609], dtype='float32').reshape([1]),
                paddle.to_tensor([0.04495077580213547], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4fca62102c0df3fbb03791c8d23120eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6127894520759583], dtype='float32').reshape([1]),
                paddle.to_tensor([0.11855943500995636], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7c2b8e66189f6917b95ed551e28bb96c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6118536591529846], dtype='float32').reshape([1]),
                paddle.to_tensor([0.08503968268632889], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4a86af379d1a028bca1ab3dd60a8d73d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.9667161107063293], dtype='float32').reshape([1]),
                paddle.to_tensor([0.06937023997306824], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f7fac62e414dd015b9b0171963f71e00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.6435269713401794], dtype='float32').reshape([1]),
                paddle.to_tensor([0.30397114157676697], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d5c501550b38ef11b6e85cdef4361f77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.7003757953643799], dtype='float32').reshape([1]),
                paddle.to_tensor([0.4753126800060272], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0333036b50b2cc2fd4f3f3c160f77e30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.7971274256706238], dtype='float32').reshape([1]),
                paddle.to_tensor([0.21060983836650848], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_82b013264839fc0dd10abe5553cb12b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdaf0d76f93adecb4d9c357efe707c8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d56bdf6e90ba527e516774c25493e003(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc98f357670e36e8f78f5eef62348906(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e67beaf329dd26bd914a26039b9e45eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.27113786339759827], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e67beaf329dd26bd914a26039b9e45eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.27113786339759827], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_05747d74306142d80cba34abe3134b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9af3a42552b8d502ed34368ba952af0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.19904105365276337], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_94485c18ba5a4dd62a245ee0d72981dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f118827a835ae8d22d9b47b7c741524(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_879b6d9f9a8e369cd0654e362e1cbabe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_879b6d9f9a8e369cd0654e362e1cbabe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_879b6d9f9a8e369cd0654e362e1cbabe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_879b6d9f9a8e369cd0654e362e1cbabe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_879b6d9f9a8e369cd0654e362e1cbabe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7306f5f9f1658980852b23ed31121c14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7306f5f9f1658980852b23ed31121c14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_879b6d9f9a8e369cd0654e362e1cbabe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_733b0c227453319d928c88572f456177(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b896e16d3f91af598704bb9d6e7b0651(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.4964856207370758]], [[0.40626269578933716]], [[0.46424606442451477]], [[0.3560028672218323]], [[0.10537204146385193]], [[0.3426622450351715]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_d2c650ff05b7c26b907b5a1f34e28fd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bc8df9c519ff817b48633b27e81105f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bc8df9c519ff817b48633b27e81105f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bc8df9c519ff817b48633b27e81105f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bc8df9c519ff817b48633b27e81105f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bc8df9c519ff817b48633b27e81105f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d83a681fff5cd996eda53d23cc7d2f33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d83a681fff5cd996eda53d23cc7d2f33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bc8df9c519ff817b48633b27e81105f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9681865d22148a9f2725350d594b66d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfb775dc3765879bc5fd27f7123e3026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49f77ebf8b2ecd03ebe00d2ae9ea7d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3f42cb838e3e31cd0f78cead9a63cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3805851b5f0323ffd727383c02ae1f9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4964f117f283b77202fba279ceb54c10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00e19d52855f7851aab7b1284012a5d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1414ea58c0f84f3390cbc2ac45eac980(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cc193b3e3925428dfac28c1f431ef87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 24, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54ae71bf3d96f1a3946996e8a1b5826b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7403ccec4d994af961219eb80d66c410(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8200b69d2e4334a5efb95c7baa41eb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1b6b9a3221179f3fa5358fba8c3fe30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9c1244c7e2b98e76422c38fefa81644(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_587951ef02534dffb31952fc0224430e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea5bbaacf1383dc473a9b13e3213eb3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8575fe42e0a8b730d655ac33dcb7281d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8575fe42e0a8b730d655ac33dcb7281d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fe59bc00f125c6744e490bf04589ff1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.48570457100868225], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a29397effd1f4367a2d7a3e7c3254761(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558a0aab38432a7e8c8fac4f040fb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8592aad28245ad3395a80a63ca63c3bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e632c1561862f033ccd9223258f04f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2e17128c8d37f6aa453b30547466c17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00a25c7062298fe912cd839327376bb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00a25c7062298fe912cd839327376bb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00a25c7062298fe912cd839327376bb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e1798cb5051f7421735d07d79d1f27f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8a3219fc1b0dbee34003e6bc78b169d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eefba6d6dd56eb4581ba8639585abea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5eefba6d6dd56eb4581ba8639585abea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fee4d718992b4a1dd9f6f71de646908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.09488201141357422], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_91ee3935b0dd9c4678d436d4fb94fd0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.06572425365447998], [-0.017537042498588562], [-0.16134634613990784], [0.1909516602754593], [-0.2708263397216797]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-0.2624059319496155], [-0.25955161452293396], [-0.3877258896827698], [-0.1352548599243164], [-0.3994396924972534]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_73b566af1e0dfb5395939469b1d7003c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.02962172031402588], [-0.012923628091812134], [-0.3421880006790161], [0.09488201141357422], [-0.03534042835235596]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1962345689535141], [-0.3319106101989746], [-0.016855984926223755], [0.2746593654155731], [-0.3117551803588867]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_a6a2951b70cf9e4e54bc43644c6c6a86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.022850245237350464], [0.2385021597146988], [-0.08091486990451813], [0.1909516602754593], [-0.03534042835235596]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1962345689535141], [-0.25955161452293396], [-0.016855984926223755], [0.30179738998413086], [-0.3117551803588867]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_783d21386e7dc842ee0f681311978813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7636e3039adc98183a222310bf78b0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4005175d5e956b664d3b4dd317a6762a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75b8b4a59de1678d4ce248e57c6ee172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e65f90cea937383335fc1882a94c8c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c83adcd7c8d07451e3952774db8d82b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4739fd4acd92ca666d6947fb0cf4a775(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ec38bfc95a9a48f223f6f8f1911beb2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_909e43114f49865e045ae33feb9ee4a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 76, 116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3024537265300751], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_11feeb27acb1b986a3cc7da17785444e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92d1865c6c3fe7c70a22221e8e302122(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97adb54098e31463adfb8da6ffe8ca30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa6a32ee5204b6f58845203c5ada7f49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c3452bec4d005ac29b2d83b6508a8b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c871c95b04760b10ac9e6d887d36cae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_083b0abe5262df9e8fe1b52981200950
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3566078841686249], dtype='float32').reshape([1]),
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc78091dac342f6b18d3282d06c8427c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc78091dac342f6b18d3282d06c8427c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc78091dac342f6b18d3282d06c8427c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f36410addd6e1a5502740935d35c2fe5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c157613ca66ec0021fe65b025f03e632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1248, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1882ba04b5c29895129733ea702d0fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9366a66defc2d7731e4e1af2c7e37d6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.010624636895954609], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9366a66defc2d7731e4e1af2c7e37d6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.010624636895954609], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a813e88aff6240a5f49bcb0b09a336b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_935113cada714865fca586b3822357fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3146562874317169], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1956dc42971d4c5618cbcff617591ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95d76a7d0da3ab0f331c1cb997e256c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc0de0f0910a11b00bf143b1b4cc3c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f1095f872d4739d8f428ca1dfdef08f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd7f181b1ce63ff4826391dd0723be99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_731168de48baaad80b466fd9e7882129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 5, 8], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.37174689769744873], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_16204aae0401132ebfd45fd0f81143d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5ba26e4e2dd99bf5f0001b5c15c0669(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_958cc55a44e7a8cc7cf0dd55da27f9bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8decfe84072275269f5a49c105c7e328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8decfe84072275269f5a49c105c7e328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8decfe84072275269f5a49c105c7e328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8decfe84072275269f5a49c105c7e328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8decfe84072275269f5a49c105c7e328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d68432761aed2ef393ddaf05fc819086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d68432761aed2ef393ddaf05fc819086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8decfe84072275269f5a49c105c7e328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61cef7b5f672c90e47d0be3258e9fa53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f14382199d60a18f439f3992e9127c45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbff8b8cd3c3dc90635503a17142976c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbff8b8cd3c3dc90635503a17142976c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbff8b8cd3c3dc90635503a17142976c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbff8b8cd3c3dc90635503a17142976c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbff8b8cd3c3dc90635503a17142976c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f14b408308dbb5119c7947e6c5541155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f14b408308dbb5119c7947e6c5541155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbff8b8cd3c3dc90635503a17142976c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e43197da007b8a1c58e634214d734c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e43197da007b8a1c58e634214d734c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e43197da007b8a1c58e634214d734c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e43197da007b8a1c58e634214d734c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e43197da007b8a1c58e634214d734c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2124f75659af044ac5b1cc0ea2fe7cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2124f75659af044ac5b1cc0ea2fe7cf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e43197da007b8a1c58e634214d734c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_608e55ccf5f3ca622b87cd857add3ea7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2106606662273407], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_608e55ccf5f3ca622b87cd857add3ea7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2106606662273407], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_97b99998a9d9d795ca6056a189e41ac8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4d5e5f0ff41297d7592bdd360c3e503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.3481174111366272], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_836569687a3f548e0f73b9c9bd4b23f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 156, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_779a82c6fe5dd2066233114781f0034d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.8479311466217041]], [[0.8519147038459778]], [[0.8904101252555847]], [[0.8855916857719421]], [[0.8917824625968933]], [[0.8464255332946777]], [[0.7774280905723572]], [[0.8978089690208435]], [[0.8423903584480286]], [[0.7676694393157959]], [[0.892359733581543]], [[0.8445429801940918]], [[0.8606671094894409]], [[0.7275188565254211]], [[0.8979865908622742]], [[0.8983840346336365]], [[0.8124725222587585]], [[0.8320170640945435]], [[0.8500043749809265]], [[0.9105454683303833]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_7b4b98da6beb069b3550e1910548f4ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9838b8f746e2753c8543b596ebf94f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40b8070a3da2d22bec8620ee20e799d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa771197311f58767836c2e15e589e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8891e681dcb44fd5b9e1cffc1f609406(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c14f62d4df17cfaacc78f0dac86ab557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be2c3d67b11b74325bdb5e4d1a169403(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d0a2a00b7d700f782e87dac399b2036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e322cbdf55112d072bc32520054f0ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]]], [[[0.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]], [[[1.0]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_c31495768bbddef1674e461a8747ce2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8827fcf835fe5d7a29c38e3a694bdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_413100995a60bfc2b8787e6dc5cda558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_083b0abe5262df9e8fe1b52981200950
        def get_inputs(self):
            return [
                paddle.to_tensor([0.19080998003482819], dtype='float32').reshape([1]),
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85d8801fd6c5fff59f31bcecc13ffa08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f758ebd7adf6e4e004f5d387ebf8fa94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d597a3f93a8c0e273eb0b60ed85ba53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b0c003872f8e01f4b643e93ebea1ca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9681865d22148a9f2725350d594b66d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49886b2f047a0d7317e66c8753b6fbf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 872, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a62a33ff5c69d19150a4df8859ba64c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01a55d353aaf2e48e8061a455f9e6e3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_250fdafe45c6c69038b4a3d93640db45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72754127e934ae861c87666a7f44c8ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0987c297c3be6c0b50c1933d0e7f5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd39fc33bff4b4cf45d9beaf52c478fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([145, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82d881ec5c01b23301b765c17b92cf1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.1887969821691513], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_72dddf1e37106d22a3e487b54791626f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c7fcf5b6ffa2a1c85464d8f125a8592
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[0.4264984428882599]], [[0.44378095865249634]]], dtype='float32').reshape([2, 1, 1]),
            ]


    class TestPrimitiveOp_d001302738fa27e7288ab06508f53f7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82a882d5b19a2b4048b94dff20ebb53c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([171, 36, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2489c342d2c614d364210539372719a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35d53efd78c539c3497f3a1654b617c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efcc4b69acdb9138757a70e64afc85f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_230df67b21b842de7dbd4e335a9946d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d075c1e25c01af95fee353533de0216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46ee9b143e0aa7f411ae14cf933c5b6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c96d0a854754e3e96cd56a830daf4fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3160fbee4122ede4693b5231d86ed8e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 256, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eedf8855fb8a6d2917f6bc478d083426(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([2.2074105739593506, 2.0814871788024902, 2.0127875804901123, 1.9578313827514648, 2.0946848392486572, 2.2787415981292725, 2.061972141265869, 1.993729829788208, 1.838011384010315, 2.260199546813965, 2.0789196491241455, 2.2942206859588623, 2.0726144313812256, 1.9446386098861694, 2.2542762756347656, 2.0304219722747803, 2.349889039993286, 2.2931485176086426, 2.0853161811828613, 2.083437442779541], dtype='float32').reshape([20]),
                paddle.to_tensor([0.9763298034667969, 0.9317722320556641, 0.9069538116455078, 0.8245687484741211, 0.909284234046936, 0.5500822067260742, 0.9664220213890076, 0.7596622705459595, 0.7154883146286011, 0.8534694314002991, 0.6166922450065613, 0.596886396408081, 0.5243674516677856, 0.7685458660125732, 0.559390127658844, 0.5170217752456665, 0.612459659576416, 0.5005311965942383, 0.6033649444580078, 0.6565635800361633], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_ae4b4364a94b9fd736bbb16d0d420e9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([1.9475164413452148, 2.2945051193237305, 2.191253185272217, 2.280691146850586, 2.127012014389038, 2.0838818550109863, 2.1300153732299805, 2.147913932800293, 2.332383632659912, 1.930082082748413, 2.3858895301818848, 2.144876003265381, 2.0804991722106934, 2.0540261268615723, 2.0113155841827393, 2.045353651046753, 1.8870145082473755, 2.307952880859375, 2.000396251678467, 2.1889240741729736], dtype='float32').reshape([20]),
                paddle.to_tensor([0.023670200258493423, 0.06822778284549713, 0.09304618835449219, 0.1754312664270401, 0.09071575105190277, 0.4499177932739258, 0.03357797861099243, 0.24033771455287933, 0.2845117151737213, 0.14653056859970093, 0.3833077549934387, 0.40311363339424133, 0.47563251852989197, 0.23145411908626556, 0.440609872341156, 0.4829781949520111, 0.3875403106212616, 0.4994688332080841, 0.3966350555419922, 0.34343641996383667], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_05e191ff0bff976b3dac3a4832437c8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.5503146648406982, 0.5240052342414856, 0.5073482990264893, 0.5036177635192871, 0.5244043469429016, 0.547767698764801, 0.5160642266273499, 0.5076965093612671, 0.49466651678085327, 0.5529568195343018, 0.5491458773612976, 0.5585044622421265, 0.5190911293029785, 0.49248918890953064, 0.5368063449859619, 0.5094084143638611, 0.5426266193389893, 0.5751357078552246, 0.5129085183143616, 0.5299163460731506], dtype='float32').reshape([20]),
                paddle.to_tensor([0.16715207695960999, 0.4314711391925812, 0.16430820524692535, 0.42761585116386414, 0.2671683132648468, 0.07692061364650726, 0.2710318863391876, 0.20835819840431213, 0.1385306417942047, 0.040651559829711914, 0.15243127942085266, 0.08873989433050156, 0.46281492710113525, 0.10008900612592697, 0.24598225951194763, 0.2274261713027954, 0.4416486322879791, 0.02435649186372757, 0.0623704232275486, 0.47828513383865356], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_bd99b2f8c18d85f7a20a2ba6664ad721(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c96a223e319353078fbd42c91f55965(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da730c72e07295ae09a61c56c468f3b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33fea203caa99affa08a1767db83f3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ae8b5f238c8569f1b8a5bbbeee79830(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_835c8ff1e1cbc1727e7bc35f26b6a233(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_379b1eb53c27c181b022935efb5f7679(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad5f675dc57222b5c6a9fde246959935(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.9418007731437683]], [[0.9117066860198975]], [[0.9077053666114807]], [[0.9410808086395264]], [[0.90910804271698]], [[0.9616879820823669]], [[0.9368215203285217]], [[0.9499552249908447]], [[0.9581537246704102]], [[0.8683320879936218]], [[0.9321792125701904]], [[0.9070103764533997]], [[0.8749677538871765]], [[0.8305680155754089]], [[0.9019274711608887]], [[0.9057850241661072]], [[0.8383494019508362]], [[0.9794363379478455]], [[0.8647102117538452]], [[0.9678527116775513]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_7b4b98da6beb069b3550e1910548f4ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9838b8f746e2753c8543b596ebf94f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f758ebd7adf6e4e004f5d387ebf8fa94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4fa183875d227eb6f2d77e2bdc6b4a0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3e88d201acae2b4ef86aa9da8b25bfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b0c003872f8e01f4b643e93ebea1ca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40babbeb95532d9254aa8aaec4896e38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60025833755fbdd51d4eb42899d9b368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cacba5e2957ab2344a4e84e0488be87e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a868d34eaf4a0b1787f8b7ad9e440cbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_250fdafe45c6c69038b4a3d93640db45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 1, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b59e3c931828067097689f1c7fb450e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d63e5039997997bb5a02c19b4d8fd8b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 336, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90bf921672694d9d40524086096930ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12a06d17ee26ce35adbfa6a1c5e3e49f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c23ee0f45525751611ca969c7c3a10d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 56, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a3feba91575df353b0f92378a6a0f54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 81], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c3632c5a14f502ad44a2e64c43959dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f3005f95d9251700a8006871c8be27f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.009707748889923096], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_9c87d1cc1c1d60edbb3c5a9fdb7ff8f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.24875834584236145], [-0.07860368490219116], [0.2262011468410492], [0.2045152485370636]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.003068208694458008], [0.1348257064819336], [-0.2989177405834198], [0.05067698657512665]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_44f9e0291852481c517c85c3c8fdbef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3149139881134033], [-0.12571868300437927], [0.0250491201877594], [-0.0416700541973114]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.1347169578075409], [0.020368844270706177], [0.3810873031616211], [-0.13437679409980774]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_c7167bab7670f480a767b177a407a0be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3149139881134033], [-0.04792898893356323], [0.2415425181388855], [0.2045152485370636]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.1347169578075409], [0.3491324782371521], [0.3810873031616211], [0.0530858188867569]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_3aaa131d74c811d744b7a6743ff16368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7e0dcce7f6803204f6cdcddc95fe1ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_faa689f9497b967e0f4f3660c30fe6fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1716bf504b05e43b14a71ae0f3a87995(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c3632c5a14f502ad44a2e64c43959dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_555b6bb06a03fe163a72b25d518f05d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_818a0efb5f245a72f2bcd37aee51f281(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49495a9aa3ea3180b5b404a12da3fcb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc5ff844347edfc2d54ea39ff37f4192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b661ba6338d07d27001e75dfb07af06b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4bce8125a614659dd86d373d355aaa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4bce8125a614659dd86d373d355aaa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4bce8125a614659dd86d373d355aaa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4bce8125a614659dd86d373d355aaa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4bce8125a614659dd86d373d355aaa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7485706caa34c1fbd1f3ab6992de41a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7485706caa34c1fbd1f3ab6992de41a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4bce8125a614659dd86d373d355aaa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28047d860f494fad10147fe82ee4ade1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a2f2c092b2bc9be5d7326c63cea806c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0353e6e7737b30d1ada9dafcef3d99c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b732131d252a4370c9634b6c3e8ac4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97ea94cf5d47951b582b1f2391f847dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0f4de2465b4ff2833dd6525a68495d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4b568a66e2145cac52d2a01856d7880(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_846ddda6516ba2355addf7aeb8740e8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_083b0abe5262df9e8fe1b52981200950
        def get_inputs(self):
            return [
                paddle.to_tensor([0.09649503976106644], dtype='float32').reshape([1]),
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0c3e9c265acbc5eace038f6f1a099b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 120, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b751c052ce32ad2d2c0941fd42f15a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 1, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_731eff189085a2520dc4349f2c754008(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.1989731639623642], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_731eff189085a2520dc4349f2c754008(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.1989731639623642], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_112be3c80bc5cf8c205ce1895e003610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d7d137279b6679e247d37c7d3e37f2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.44187647104263306], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6cd3930a2d4ca41327efda396d638d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 200, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e22dfd216974eb0d53912a3383bb7070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 400, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bd3860225fcd2422216a69ab07da1d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d2b6b5e12caf0d5d6223db21e514af8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7602f0f687fbf35cda11932c70b9466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12a200176ee6078e3bf24c3bf61f0aaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43841ac51164a5899cb2a0d636bafb33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2e17128c8d37f6aa453b30547466c17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 1, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d68490d1cba3c97f1b6dd517d036b51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 100, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62c4aa44126a34ad4951bc015d9a76c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48a48fbea25565f027d8b453f10ca56e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46ee9b143e0aa7f411ae14cf933c5b6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dc369f4ba1bfe7d722a6659a709c4b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7ae8ac84a294bd7d8192ca6aa89c4d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3acdf326cd11b6ef38855cf30f8fa6a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a71e9378acb365d8745a3d507866556(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75b66f74db17c42f4d36f8fda91254d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 320, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc78091dac342f6b18d3282d06c8427c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ded139b4ebef7176e23c4f2052415d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c17ded5975467a20bd1b677301db9b85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 38, 58], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.026828795671463013], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fc5f75ce8b6646aeb8e7f86217a99fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bfabe7a3c8f0e40f76c77d08e39adea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aed2fdbf60b5583109e21a63f88bfd1b
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d8aae3210ee775997549da34aad7260(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.2093075066804886], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b98ae46da5b331d78dc9ebb4ab1debfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8339b99cfe7fc4f043b64d9d822e603e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46ba60237f4f37a6a3f0d0a0dcdb5ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a37c56350548dc1f01aece19a45b080(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 19], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.34211570024490356], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f14382199d60a18f439f3992e9127c45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1eb4a3a0137c7e364b86407d0bd68280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30e19cec19a3c71779868f2938af7ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90bf921672694d9d40524086096930ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5269b4fc0b893944fd82cd0839d9406(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34c2f9f32c19fbd2bc9cc04bda508061(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc575180015122180832cdd43ac99ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_185f02712f544d57c599e51e57ca85e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_533b3e165484b25db24e5c73996d2707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_acf7f37b059911a0c171e74b18a511c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_fc01f473b8849dfc6ea0b392d16459ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_71eb71c4b90c532dd216bdee3e689151(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]], dtype='float32').reshape([1, 24, 1, 1]),
            ]


    class TestPrimitiveOp_26a468bc293239e876fbd941b5c41377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 11, 17], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.4490073621273041], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_893ea6e69510da8bf0f4a93d082b377f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08368d746dd71cc00c06a31aed5ae77b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d36fa7ecea096384d47700594f67ea7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b16ed01c827c2e0199463d0930a38560(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.24439510703086853], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a501d3d57b59eb0bb3b3178034e6c9bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3c29aabf761f3926bef2f75312cc94c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3c29aabf761f3926bef2f75312cc94c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3c29aabf761f3926bef2f75312cc94c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3c29aabf761f3926bef2f75312cc94c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3c29aabf761f3926bef2f75312cc94c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf3404ad982158499580266fdcdd34ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf3404ad982158499580266fdcdd34ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3c29aabf761f3926bef2f75312cc94c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d31292b5009c1a92cfeb7ff4e63492ac
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_073c33dca1dbba365c9a1688c877483e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e16afe806f018b97475625a656114954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7176b844e4fde08ff08c5f05024aa364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a40b093c5738f9329a76df9d7ea07b6f
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_164fe37a47301544a42b09d06d7eab4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 19, 29], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.11256954073905945], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_eebcf7842fec6aa14eb90a6d4b9a4d1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 624, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c01e7865e9224ff4a8b737109a8411b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2458c59f218dcf159af81ad8da3754ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6675dd8393b353de2d380e5bafc030fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55b2fbc47229eca08876cac309ce6c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 72, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2489c342d2c614d364210539372719a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_231a795a24b8177947ba7979a7ea3202
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 1, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_046e95f66d670abfbdeb9cb932709610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd5e539bfd4ce01925acc95da86439e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12319e55a0540b09ebcaadf571b1b8e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d153b0d295d7f9f20a17d784ed2b41cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.33400335907936096], dtype='float32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()