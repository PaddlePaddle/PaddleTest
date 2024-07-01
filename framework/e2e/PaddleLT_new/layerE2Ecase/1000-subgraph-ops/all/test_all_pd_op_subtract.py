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
    class PrimitiveOp_0a189c700f923c014e513b319e236e8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee81927185378170745721caa17cd437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.05118778347969055]], [[0.10903573036193848]], [[0.4847700595855713]], [[0.2270936667919159]], [[0.4217580258846283]], [[0.2526901066303253]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.5237996578216553]], [[0.6988669633865356]], [[0.5810272693634033]], [[0.6514535546302795]], [[0.5936262011528015]], [[0.7933114767074585]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_df770788cc453d2632a17c13e6851446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.41506168246269226]], [[0.11528967320919037]], [[0.15182477235794067]], [[0.2979087233543396]], [[0.05990821495652199]], [[0.09339520335197449]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7646562457084656]], [[0.5594422817230225]], [[0.608974814414978]], [[0.7140263915061951]], [[0.7158714532852173]], [[0.7036947011947632]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_86702e1b01684d81107fc9d2e782dd77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6084071aa209c01df02838230696ee7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_263baae981c6452c4fc38dfdeb3d1bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6945f1db5df5b2f8eb987ce6f1638184(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0da04656b12eb2bd7976352419398175(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12096, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37be542e9a3916e40a2b656d9dc34c2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0da04656b12eb2bd7976352419398175
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6133dae93b7d7a9e251cfe215b215e67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cabd32ccc91f955e85c15f279d1c714a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.22274713218212128, 0.11812928318977356]], [[0.40234601497650146, 0.32135093212127686]], [[0.24842727184295654, 0.3543384373188019]], [[0.4109976589679718, 0.028806988149881363]], [[0.09972511976957321, 0.48537272214889526]], [[0.24782226979732513, 0.12431900948286057]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.09539089351892471, 0.014189804904162884]], [[0.4132586717605591, 0.26804450154304504]], [[0.2966609001159668, 0.13498319685459137]], [[0.1327727735042572, 0.43529269099235535]], [[0.22661817073822021, 0.15752831101417542]], [[0.050448279827833176, 0.11166179925203323]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_9b37f421d7cc7a6e84002376f8451168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.025420304387807846, 0.09367721527814865]], [[0.14023251831531525, 0.34991103410720825]], [[0.418991357088089, 0.374828040599823]], [[0.07487571239471436, 0.2823382019996643]], [[0.3401561975479126, 0.3036860525608063]], [[0.23089678585529327, 0.15044544637203217]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.09539089351892471, 0.014189804904162884]], [[0.4132586717605591, 0.26804450154304504]], [[0.2966609001159668, 0.13498319685459137]], [[0.1327727735042572, 0.43529269099235535]], [[0.22661817073822021, 0.15752831101417542]], [[0.050448279827833176, 0.11166179925203323]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    
    class PrimitiveOp_c7df6014de2cb40241de025d80c39dcd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c35797f7113b36732ed94ce53ebca0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7df6014de2cb40241de025d80c39dcd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.2746506929397583, 0.3692845404148102]], [[0.2529679834842682, 0.3538026511669159]], [[0.1918400079011917, 0.46419987082481384]], [[0.4446069300174713, 0.48151710629463196]], [[0.22252815961837769, 0.22237402200698853]], [[0.07176125049591064, 0.38622698187828064]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_97bb131ea97a8cf0466245328fb49165(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17ea37bfbf76b52dc3a6ec3dffd93937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.37460729479789734, 0.4844105839729309, 0.181888610124588, 0.44207683205604553, 0.02534153312444687, 0.3211316466331482, 0.009519957937300205, 0.40302348136901855, 0.22982287406921387, 0.10425077378749847, 0.3179247975349426, 0.40923941135406494, 0.2358681559562683, 0.14795830845832825, 0.06801445782184601, 0.4440118670463562], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_b7f5cc56867c835ed114d2cd9efcb717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37460729479789734, 0.4844105839729309, 0.181888610124588, 0.44207683205604553, 0.02534153312444687, 0.3211316466331482, 0.009519957937300205, 0.40302348136901855, 0.22982287406921387, 0.10425077378749847, 0.3179247975349426, 0.40923941135406494, 0.2358681559562683, 0.14795830845832825, 0.06801445782184601, 0.4440118670463562], dtype='float32').reshape([16]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_efe8d2d8a73054c00e916a968eb0939f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300], dtype='float32'),
                paddle.static.InputSpec(shape=[300], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_242d399e131f904b2a97ae2bb781ad1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_efe8d2d8a73054c00e916a968eb0939f
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_242d399e131f904b2a97ae2bb781ad1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_efe8d2d8a73054c00e916a968eb0939f
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b02f42b1db1ca1710e95443e70c4989(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36d070ac6265170c6fe3db6a8e299e6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7c93cb758f546807fe57e0c4dcc0d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d71057689b12f3a5555e946fa88155c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_599b3dfefba3119bcb4ff25be431050b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36d070ac6265170c6fe3db6a8e299e6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f43ab2165b83b58c7ab2b262e252c0b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17805549502372742, 0.08694770932197571, 0.24425704777240753, 0.132246196269989], [0.13605213165283203, 0.3651181161403656, 0.3882414996623993, 0.2905106246471405], [0.41753944754600525, 0.4236792027950287, 0.3029243052005768, 0.4083331525325775], [0.44502976536750793, 0.0707036629319191, 0.4251031279563904, 0.20904578268527985], [0.28272974491119385, 0.3605213761329651, 0.46954378485679626, 0.444469153881073]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.12191290408372879, 0.35174989700317383, 0.10305600613355637, 0.01692168414592743], [0.1374139040708542, 0.3359614610671997, 0.18675795197486877, 0.011088688857853413], [0.31233879923820496, 0.19828705489635468, 0.09374342858791351, 0.2324703186750412], [0.09410864859819412, 0.44495889544487, 0.17497994005680084, 0.22089767456054688], [0.315926194190979, 0.4338832497596741, 0.2439688891172409, 0.04640224948525429]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_62636f2ca6ea405532c01f541c843047(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 5376, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5eed7aeb83ace981c449bf47a3a72522(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62636f2ca6ea405532c01f541c843047
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2353b992d4f1334cc422450b6a9cd76b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03461017832159996, 0.0004527518176473677, 0.36749422550201416, 0.32795700430870056], [0.3849736750125885, 0.032286059111356735, 0.061482444405555725, 0.3820096254348755], [0.102759949862957, 0.4908919334411621, 0.49278682470321655, 0.4886923134326935], [0.3849736750125885, 0.032286059111356735, 0.061482444405555725, 0.3820096254348755], [0.102759949862957, 0.4908919334411621, 0.49278682470321655, 0.4886923134326935]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.004080449230968952, 0.28355368971824646, 0.0013239141553640366, 0.3698040843009949], [0.45567235350608826, 0.2550033628940582, 0.02040109969675541, 0.4188835024833679], [0.48355358839035034, 0.49027279019355774, 0.13922299444675446, 0.3933846354484558], [0.45567235350608826, 0.2550033628940582, 0.02040109969675541, 0.4188835024833679], [0.48355358839035034, 0.49027279019355774, 0.13922299444675446, 0.3933846354484558]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93789134cdc7acce06a130729c021eb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13368675112724304], [0.3421638607978821], [0.3978815972805023], [0.15133216977119446], [0.22187696397304535], [0.2794041037559509], [0.01730276271700859], [0.08600067347288132], [0.0719427838921547]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.38795778155326843], [0.4817730784416199], [0.1863391399383545], [0.44741085171699524], [0.29538026452064514], [0.3727307915687561], [0.4028562605381012], [0.3053695261478424], [0.31240081787109375]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2f6f5d7a3ec320339490b32f1c85cb5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16374197602272034], [0.18657687306404114], [0.13615421950817108], [0.26141688227653503], [0.006397350691258907], [0.38319242000579834], [0.3071695864200592], [0.004688146524131298], [0.23858243227005005]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.49473410844802856], [0.3323400914669037], [0.3997913599014282], [0.4925849139690399], [0.17630773782730103], [0.12873531877994537], [0.08800354599952698], [0.43823152780532837], [0.4259171187877655]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_bc1739b7884f8088f9636cfccb2f05bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13368675112724304], [0.39139324426651], [0.3978815972805023], [0.2982294261455536], [0.2754661738872528], [0.3869280219078064], [0.10337948054075241], [0.465537965297699], [0.4861213266849518]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14307871460914612], [0.19619788229465485], [0.1863391399383545], [0.2124066799879074], [0.035387687385082245], [0.1282057762145996], [0.4028562605381012], [0.007067443337291479], [0.31240081787109375]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_db56da587f2c027e1e7880096e1eeaa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16374197602272034], [0.22206810116767883], [0.18589580059051514], [0.26141688227653503], [0.006397350691258907], [0.38319242000579834], [0.3071695864200592], [0.40571993589401245], [0.43421873450279236]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14178644120693207], [0.06213818117976189], [0.3997913599014282], [0.11639508605003357], [0.015070164576172829], [0.12873531877994537], [0.05166096240282059], [0.269934743642807], [0.09819310158491135]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_8930c125f87ae71f9c24f2a6c8b43ad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2243945300579071], [0.3421638607978821], [0.4742037057876587], [0.15133216977119446], [0.22187696397304535], [0.2794041037559509], [0.01730276271700859], [0.08600067347288132], [0.0719427838921547]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.38795778155326843], [0.4817730784416199], [0.017573527991771698], [0.44741085171699524], [0.29538026452064514], [0.3727307915687561], [0.023896757513284683], [0.3053695261478424], [0.20216022431850433]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_ad3cd92c1671398a40ab6e3d546b6c74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4538259506225586], [0.18657687306404114], [0.13615421950817108], [0.37509968876838684], [0.4772617220878601], [0.39100539684295654], [0.4677458703517914], [0.004688146524131298], [0.23858243227005005]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.49473410844802856], [0.3323400914669037], [0.2790012061595917], [0.4925849139690399], [0.17630773782730103], [0.10014522075653076], [0.08800354599952698], [0.43823152780532837], [0.4259171187877655]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_b9cd13bac2d0d51e05320a707c3ad954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.006484865676611662], [0.05156746506690979], [-0.11047624051570892], [0.0472310408949852], [-0.024203266948461533], [0.03868870064616203], [-0.07902292162179947], [0.15735942125320435], [0.08276878297328949]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_81949986093347ad04de44bf5971891b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2243945300579071], [0.39139324426651], [0.4742037057876587], [0.2982294261455536], [0.2754661738872528], [0.3869280219078064], [0.10337948054075241], [0.465537965297699], [0.4861213266849518]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14307871460914612], [0.19619788229465485], [0.017573527991771698], [0.2124066799879074], [0.035387687385082245], [0.1282057762145996], [0.023896757513284683], [0.007067443337291479], [0.20216022431850433]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_c2d4f9c220c884c758de11a51ab21aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4538259506225586], [0.22206810116767883], [0.18589580059051514], [0.37509968876838684], [0.4772617220878601], [0.39100539684295654], [0.4677458703517914], [0.40571993589401245], [0.43421873450279236]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14178644120693207], [0.06213818117976189], [0.2790012061595917], [0.11639508605003357], [0.015070164576172829], [0.10014522075653076], [0.05166096240282059], [0.269934743642807], [0.09819310158491135]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9ab8f63c2a091889b7eb6088fc1ffcdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.02537374570965767], [0.031217578798532486], [-0.04251473769545555], [0.022202739492058754], [0.11096224188804626], [0.0752519965171814], [0.03307155892252922], [0.06225350871682167], [0.09541821479797363]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.006484865676611662], [0.05156746506690979], [-0.11047624051570892], [0.0472310408949852], [-0.024203266948461533], [0.03868870064616203], [-0.07902292162179947], [0.15735942125320435], [0.08276878297328949]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_215345af8b85a7022f2700203001b9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [-0.0], [0.0], [-0.0], [0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.7444261312484741], [-0.6518726348876953], [-1.5985398292541504], [-1.1272618770599365], [1.2181216478347778], [0.4858780801296234], [3.3894524574279785], [-1.5277197360992432], [0.13256831467151642]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_038caaf8caddf0ed66edf1b7a16c2165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7216cb943e51a117d96edb4973fa4f72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.18063703179359436]], [[0.2793530225753784]], [[0.04766196757555008]], [[0.008773964829742908]], [[0.3327653408050537]], [[0.4748668670654297]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6437686085700989]], [[0.66782146692276]], [[0.7871782183647156]], [[0.6366011500358582]], [[0.5327322483062744]], [[0.6649988293647766]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_065049d6aa2dcbe354a6cfa9b103e62f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.27055323123931885]], [[0.030705591663718224]], [[0.3505762219429016]], [[0.4922664165496826]], [[0.22030052542686462]], [[0.06326745450496674]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7747390270233154]], [[0.5769653916358948]], [[0.551421046257019]], [[0.8195163011550903]], [[0.7298972010612488]], [[0.6213920712471008]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_13e483f6693cab66aa39f85c3d01bbda(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1511c58fb8fb90d8e4ee1e0793577549(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13e483f6693cab66aa39f85c3d01bbda
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f89cb66406c9afe070b3a6b95c177ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c524e16872b47744e49dce4b24a6b012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79d849c3be33c301145579fc57ad72db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa254b4c474f9e2e1154af56fb39774e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f89cb66406c9afe070b3a6b95c177ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_498f8b9df0aa81f47d26c72bd312f338(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16312061250209808, 0.039097484201192856, 0.31288108229637146, 0.4763258099555969], [0.4967135787010193, 0.12254718691110611, 0.134210467338562, 0.22143881022930145], [0.33521342277526855, 0.3220093846321106, 0.48359137773513794, 0.09311709553003311], [0.4967135787010193, 0.12254718691110611, 0.134210467338562, 0.22143881022930145], [0.33521342277526855, 0.3220093846321106, 0.48359137773513794, 0.09311709553003311], [0.11703887581825256, 0.3053649067878723, 0.2134171426296234, 0.02529965713620186], [0.11703887581825256, 0.3053649067878723, 0.2134171426296234, 0.02529965713620186]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.1903732866048813, 0.4341961145401001, 0.336517870426178, 0.34890449047088623], [0.29464438557624817, 0.1553640365600586, 0.4579049348831177, 0.3192165493965149], [0.061085209250450134, 0.1616355925798416, 0.03644360974431038, 0.09199950844049454], [0.29464438557624817, 0.1553640365600586, 0.4579049348831177, 0.3192165493965149], [0.061085209250450134, 0.1616355925798416, 0.03644360974431038, 0.09199950844049454], [0.46255189180374146, 0.1523735076189041, 0.4603758454322815, 0.29759618639945984], [0.46255189180374146, 0.1523735076189041, 0.4603758454322815, 0.29759618639945984]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_d3d1358f2a2a08bd6e088e7a1eeb4499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3d1358f2a2a08bd6e088e7a1eeb4499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8f9019af5d1b9974919cc09d4a01956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7ad4ff1c512290975badd6950da16fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a157008b124ce5baf968e62389b177c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45227816700935364, 0.2502661347389221, 0.16833464801311493, 0.15319602191448212, 0.18911296129226685, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_223fa79905c437c3155e1b451cee46f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.27197179198265076, 0.4429328143596649, 0.17026685178279877, 0.08001313358545303, 0.23915867507457733, 0.05345135182142258], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0c4289316532007c34eab4af1235391c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.22168461978435516, 0.48501694202423096, 0.36503466963768005, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146], dtype='float32').reshape([6]),
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.22442752122879028, 0.4289640784263611, 0.301647424697876, 0.0385211743414402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c40ae14ea19d06f1d9002ddc3aff5dbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3964419662952423, 0.30897533893585205, 0.40568625926971436, 0.10646888613700867, 0.4039801359176636, 0.40599721670150757], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3058600425720215, 0.14679817855358124, 0.3744381070137024, 0.0002531889476813376, 0.2294905185699463, 0.37590494751930237], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d2d904ca7c2ab8a7207e6a85f2521c6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.22168461978435516, 0.3048173189163208, 0.36503466963768005, 0.15319602191448212, 0.32081952691078186, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.39980247616767883, 0.4289640784263611, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4ea3dfb939dd636f6f5420582e9d1e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.30897533893585205, 0.17026685178279877, 0.10646888613700867, 0.4039801359176636, 0.05345135182142258], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3236333429813385, 0.14679817855358124, 0.3744381070137024, 0.4712778925895691, 0.4077642858028412, 0.37590494751930237], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f0018c260eae78b633642ceb9cab0603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45227816700935364, 0.3048173189163208, 0.39980247616767883, 0.15319602191448212, 0.32081952691078186, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_275571931a658b2b5045df1fc385ee20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.4429328143596649, 0.17026685178279877, 0.4712778925895691, 0.4077642858028412, 0.05345135182142258], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_20b79edfd5ffe489967c0d0123d4fba4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02505781129002571, 0.017090944573283195, 0.004393713548779488, -0.0008124950109049678, 0.017373912036418915, 0.0037726969458162785], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0, -0.0, 0.0, 0.0, -0.0, -0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_380f33707b7ae5cd8f494a176a384d65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3431909680366516, 0.27754172682762146, 0.2840685546398163, 0.12728255987167358, 0.25496625900268555, 0.12390746176242828], dtype='float32').reshape([6]),
                paddle.to_tensor([0.360000342130661, 0.43232473731040955, 0.294731080532074, 0.42513933777809143, 0.3514323830604553, 0.09111270308494568], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_852829e2715d76210d3c8a97cd9c4053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.29780256748199463, 0.2295193076133728, 0.16009734570980072, 0.27564552426338196, 0.32346147298812866, 0.03871399909257889], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3511509895324707, 0.22788676619529724, 0.3900621831417084, 0.053361035883426666, 0.31673532724380493, 0.39095109701156616], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_51e7df6c353abe09be8897f41b27ada3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45227816700935364, 0.48501694202423096, 0.39980247616767883, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.22442752122879028, 0.10136908292770386, 0.301647424697876, 0.0385211743414402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4f36692060e295d42634bd2cb1932b5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3964419662952423, 0.4429328143596649, 0.40568625926971436, 0.4712778925895691, 0.4077642858028412, 0.40599721670150757], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3058600425720215, 0.01610579900443554, 0.14992783963680267, 0.0002531889476813376, 0.2294905185699463, 0.023976648226380348], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_bef46a1c9a587ae7c8377e34146bbb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.2543535232543945, 0.5762419700622559, 1.3521130084991455, -0.07189424335956573, 0.5185477137565613, 1.2921454906463623], dtype='float32').reshape([6]),
                paddle.to_tensor([-1.33828866481781, -0.12711717188358307, -1.4831517934799194, -0.13169337809085846, 0.6631419658660889, 0.6102384924888611], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_b9d15ccce6c65cccd5fa153803ace54a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5290111f4397677d9e72c8d15f0e5924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d71057689b12f3a5555e946fa88155c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_599b3dfefba3119bcb4ff25be431050b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9d15ccce6c65cccd5fa153803ace54a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_420658d9f2294004a5766f03ff5fd9e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8400, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38f202d3c44f05c564571b58a18d01f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420658d9f2294004a5766f03ff5fd9e3
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e74a419d8dcd91fc0129f1f725790e44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.18319472670555115, 0.21686814725399017, 0.4732165038585663, 0.03563392162322998, 0.26207268238067627, 0.130903959274292, 0.44992533326148987, 0.0970231369137764, 0.4192184805870056, 0.3830379247665405, 0.41160717606544495, 0.21541158854961395, 0.13209235668182373, 0.42042574286460876, 0.40241438150405884, 0.11916881799697876, 0.17227095365524292, 0.34461653232574463, 0.4960813820362091, 0.1515713632106781, 0.1289307177066803, 0.3049981892108917, 0.004009498283267021, 0.4341105818748474], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_0119d8a5aad78c47aba505cc0d8f954e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.18319472670555115, 0.21686814725399017, 0.4732165038585663, 0.03563392162322998, 0.26207268238067627, 0.130903959274292, 0.44992533326148987, 0.0970231369137764, 0.4192184805870056, 0.3830379247665405, 0.41160717606544495, 0.21541158854961395, 0.13209235668182373, 0.42042574286460876, 0.40241438150405884, 0.11916881799697876, 0.17227095365524292, 0.34461653232574463, 0.4960813820362091, 0.1515713632106781, 0.1289307177066803, 0.3049981892108917, 0.004009498283267021, 0.4341105818748474], dtype='float32').reshape([24]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae952fa6ab94a8a65e967c6c32df0c90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4031a6eb6a1734d611559f8d3c7f466a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_945bb918603acc2712b4367c94692119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5015f5025e0809d598f06568bf7776b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae952fa6ab94a8a65e967c6c32df0c90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a6792c661b11940d4bbd9a83d82bdea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.05085877701640129, 0.22930851578712463, 0.13148881494998932, 0.16494080424308777], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_1124143059ea244197c4b0f83ac163cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.05085877701640129, 0.22930851578712463, 0.13148881494998932, 0.16494080424308777], dtype='float32').reshape([4]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
            ]


    
    class PrimitiveOp_9aefef6272755cd49e3a5bc73c4d30dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3886acca8e24a8568f34c4fba9f5def(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9aefef6272755cd49e3a5bc73c4d30dc
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_603b9ad9e6326deca0eac263f9380b59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9aefef6272755cd49e3a5bc73c4d30dc
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0094315bc00efd8603c112614d3ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.025187794119119644, 0.23439233005046844, 0.3178129196166992, 0.09868212789297104], [0.0008286791271530092, 0.3812451660633087, 0.04935451224446297, 0.14829841256141663], [0.33462032675743103, 0.184892475605011, 0.21668128669261932, 0.3788776993751526], [0.16523830592632294, 0.30108246207237244, 0.4440777599811554, 0.17766731977462769], [0.16523830592632294, 0.30108246207237244, 0.4440777599811554, 0.17766731977462769], [0.33462032675743103, 0.184892475605011, 0.21668128669261932, 0.3788776993751526]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.29314377903938293, 0.13475337624549866, 0.26757317781448364, 0.0016862310003489256], [0.49155473709106445, 0.4800690710544586, 0.006142769008874893, 0.016341568902134895], [0.11181717365980148, 0.44043388962745667, 0.00794908031821251, 0.2214917689561844], [0.08348874747753143, 0.09139897674322128, 0.14279025793075562, 0.19967620074748993], [0.08348874747753143, 0.09139897674322128, 0.14279025793075562, 0.19967620074748993], [0.11181717365980148, 0.44043388962745667, 0.00794908031821251, 0.2214917689561844]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_d13659647a3ff52ade94368a0445de32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42409706115722656, 0.487624853849411, 0.3005632758140564, 0.3299276530742645], [0.05977506563067436, 0.10115164518356323, 0.1450292468070984, 0.07699044793844223], [0.2550349235534668, 0.18359854817390442, 0.0963495597243309, 0.19211435317993164], [0.4906561076641083, 0.45334500074386597, 0.23444624245166779, 0.07382465898990631], [0.42409706115722656, 0.487624853849411, 0.3005632758140564, 0.3299276530742645]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.0006221520598046482, 0.03902745991945267, 0.32607606053352356, 0.49414271116256714], [0.4453412890434265, 0.42279648780822754, 0.0333394780755043, 0.0391206294298172], [0.2535695433616638, 0.045391522347927094, 0.02295094169676304, 0.22988493740558624], [0.1734488159418106, 0.3022133409976959, 0.45550966262817383, 0.16303986310958862], [0.0006221520598046482, 0.03902745991945267, 0.32607606053352356, 0.49414271116256714]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_7e6ac12dce03e35fbf8971aa6b9b3536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae908fea3801793f51f3c10a6d0241f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27537572383880615]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.20313051342964172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_13c8400d047220d0856726d7be87927b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14600974321365356]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.15484283864498138]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_82a82f38da0268770e1bbc84f9679476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27537572383880615]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.178619846701622]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_d370bca47285bdaf216ae2bafbd8d803(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.19937725365161896]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.15484283864498138]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_e216a54cba76f00a6b4d149cdee67861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31011876463890076]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.20313051342964172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_be06af25cbc0542d9c38c46e92090569(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14600974321365356]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.004780620336532593]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1ed01eb0bdd00df7b251bcf9491c42b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01941882260143757]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_f0af44b2d9fbc800b988476f0b8c8e8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31011876463890076]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.178619846701622]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_01071e90ddb102542b43fbd437c53988(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.19937725365161896]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.004780620336532593]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_0b2e20a998479a0a7d4e58352c26cc56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.025589246302843094]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01941882260143757]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_75d97d106cb27ed001c1a4c23fc11d07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.24113346636295319]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_6557fae0c6fc0a9827866d0538cb2208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0779203251004219], [0.256033331155777], [0.2259289026260376], [0.08137910068035126], [0.0947040542960167], [0.32221317291259766]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3637081980705261], [0.39640137553215027], [0.31833428144454956], [0.4412391483783722], [0.43500569462776184], [0.4164228141307831]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_bc5c6047c9d16af3e712e0ddc148a1ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4113561809062958], [0.32371410727500916], [0.34662380814552307], [0.24301113188266754], [0.09847179055213928], [0.15297983586788177]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.2915404736995697], [0.1776416003704071], [0.19010786712169647], [0.35384827852249146], [0.26003992557525635], [0.24949562549591064]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_5b344c820ae622df6c36df9500459e82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28181594610214233], [0.256033331155777], [0.3946605324745178], [0.4794350862503052], [0.0947040542960167], [0.32221317291259766]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3637081980705261], [0.39640137553215027], [0.31833428144454956], [0.4412391483783722], [0.11331885308027267], [0.4164228141307831]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_c3ffdbc6df3359afdc8714ec2cf4007a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4542856514453888], [0.32371410727500916], [0.47006481885910034], [0.24301113188266754], [0.31819093227386475], [0.3337242603302002]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.2915404736995697], [0.06591049581766129], [0.19010786712169647], [0.17498598992824554], [0.26003992557525635], [0.24949562549591064]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_60467d731a1556caef9060895a9da9cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0779203251004219], [0.28568974137306213], [0.2259289026260376], [0.08137910068035126], [0.37925606966018677], [0.45194125175476074]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.269661545753479], [0.24576835334300995], [0.09308407455682755], [0.3879702687263489], [0.43500569462776184], [0.14424768090248108]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_1d23722bd9948552ba27bd26b4ca1f28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4113561809062958], [0.38559842109680176], [0.34662380814552307], [0.25977855920791626], [0.09847179055213928], [0.15297983586788177]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.25507786870002747], [0.1776416003704071], [0.0561361089348793], [0.35384827852249146], [0.1015116423368454], [0.23805175721645355]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_7b699185b21847a647bc7ed68fdc0822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.043292563408613205], [-0.02788546308875084], [0.05995785444974899], [0.03143922984600067], [-0.0009129986283369362], [-0.034111231565475464]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8542b1dc1eafd7abffe407ac22e9400b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28181594610214233], [0.28568974137306213], [0.3946605324745178], [0.4794350862503052], [0.37925606966018677], [0.45194125175476074]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.269661545753479], [0.24576835334300995], [0.09308407455682755], [0.3879702687263489], [0.11331885308027267], [0.14424768090248108]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_52a4e2248f968d31941d20bbaaa27d15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4542856514453888], [0.38559842109680176], [0.47006481885910034], [0.25977855920791626], [0.31819093227386475], [0.3337242603302002]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.25507786870002747], [0.06591049581766129], [0.0561361089348793], [0.17498598992824554], [0.1015116423368454], [0.23805175721645355]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_dc9e8005470dd0488630b82991647e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.002421251032501459], [0.012762386351823807], [0.12483116239309311], [0.007755537051707506], [0.057623084634542465], [0.02943781390786171]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.043292563408613205], [-0.02788546308875084], [0.05995785444974899], [0.03143922984600067], [-0.0009129985119216144], [-0.034111231565475464]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_a23cf7642168fe1a0e9be0478771cb53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0], [-0.0], [-0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[18.880247116088867], [3.1849725246429443], [0.5196884274482727], [-3.053778648376465], [1.0158443450927734], [2.1587555408477783]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8938a22bdbc164cbf0aca9fd52ca546e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2748917043209076, 0.08581192046403885, 0.4370581805706024, 0.22711123526096344], [0.2532287836074829, 0.10648109763860703, 0.32183951139450073, 0.3253396153450012], [0.24018266797065735, 0.39429283142089844, 0.3479117155075073, 0.4558558762073517], [0.025166206061840057, 0.25497257709503174, 0.1559371054172516, 0.08968935906887054]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.30902770161628723, 0.3745928108692169, 0.090107761323452, 0.03648531064391136], [0.023343751206994057, 0.16288305819034576, 0.36057403683662415, 0.0980585440993309], [0.2143491953611374, 0.08171876519918442, 0.4514274299144745, 0.21363447606563568], [0.3628651201725006, 0.32337111234664917, 0.040081609040498734, 0.15516340732574463]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a75323c774ff55e2853da30d2527d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62e4040bd42bc4100b21448a3435cd14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_596c6c32fa9af7fcc015fedae58561c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad1ecf87ccfeb2618d3ef1463541a4ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebf0eda5c97a28b86bf8fc08b38fafd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62e4040bd42bc4100b21448a3435cd14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac684e8252a13736150650883a246c0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2859324514865875, 0.40432310104370117, 0.14047105610370636, 0.1841762661933899], [0.2859324514865875, 0.40432310104370117, 0.14047105610370636, 0.1841762661933899], [0.33153173327445984, 0.33179908990859985, 0.12240845710039139, 0.06773053109645844], [0.3055313229560852, 0.4260094463825226, 0.10106411576271057, 0.32232150435447693], [0.17913493514060974, 0.2465459108352661, 0.1178114265203476, 0.0793425589799881], [0.004713766276836395, 0.02173936739563942, 0.19002436101436615, 0.2897177040576935], [0.009822460822761059, 0.3363867402076721, 0.18706294894218445, 0.48170849680900574]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.14499889314174652, 0.40469223260879517, 0.4757552444934845, 0.10206005722284317], [0.14499889314174652, 0.40469223260879517, 0.4757552444934845, 0.10206005722284317], [0.44132161140441895, 0.005613656714558601, 0.0779615044593811, 0.04862334579229355], [0.09588927030563354, 0.43522098660469055, 0.37613382935523987, 0.46532732248306274], [0.24823467433452606, 0.010199218988418579, 0.3539154529571533, 0.3569227159023285], [0.30644461512565613, 0.336038738489151, 0.31734803318977356, 0.3516157567501068], [0.10263757407665253, 0.46752721071243286, 0.16771270334720612, 0.4427560567855835]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bf17f7c12766d8b1a40f5233d27dfb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_711054dec1761641045fdade0d16192f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13e483f6693cab66aa39f85c3d01bbda
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd93eb3c550e6e5dca0822df9e706a66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d45b3b244bdc15a67916da220388159(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2211645ab1b11819ca1eae253624ea5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0ddc0eaf16bb43fd2ad7a3644c7757b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd93eb3c550e6e5dca0822df9e706a66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a233cd8e7202fd257b91e458a940aca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e913e59f21219d317a22ba2072939329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c3f7c50469205aa8991ff5a4cd21352(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38ea7f0e6fc4386f37a8ceecbe4459dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a233cd8e7202fd257b91e458a940aca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9633b487c45d7cf36528b464d78896bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f0923bdb1c1ee9dcbeb6d6085f08a47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25175294280052185, 0.39583370089530945, 0.476584792137146, 0.11600856482982635], [0.19268378615379333, 0.33271491527557373, 0.48918667435646057, 0.00018414220539852977], [0.19268378615379333, 0.33271491527557373, 0.48918667435646057, 0.00018414220539852977], [0.42016997933387756, 0.4741819500923157, 0.3954072594642639, 0.020229332149028778], [0.31349611282348633, 0.2956351637840271, 0.3755139708518982, 0.13446329534053802], [0.004371978342533112, 0.30586355924606323, 0.08176377415657043, 0.4264273941516876]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.1815273016691208, 0.15519441664218903, 0.2377370148897171, 0.21099913120269775], [0.15358561277389526, 0.48909053206443787, 0.28568097949028015, 0.2635892927646637], [0.15358561277389526, 0.48909053206443787, 0.28568097949028015, 0.2635892927646637], [0.488042950630188, 0.4658406674861908, 0.28834664821624756, 0.2730041742324829], [0.018005892634391785, 0.48755109310150146, 0.45424145460128784, 0.039086345583200455], [0.11298181861639023, 0.40279942750930786, 0.24674753844738007, 0.23806215822696686]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c727b9dd7ef7f03d1b90627575b81b22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2e03645a6c00b3a0b03796873141d1d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 1, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef18a23c72b13c1a445b62c07440f2d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e03645a6c00b3a0b03796873141d1d0
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.5930296182632446, 5.424015045166016, 0.8112000823020935, 0.36732006072998047], [0.827667236328125, 0.2817864716053009, 0.6280726790428162, 1.0955476760864258]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7baa23aee13e28f05446515ca48869c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6069, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64754650221e9ad9afda43ce04d2b47d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7baa23aee13e28f05446515ca48869c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abe3bfd3b84c2bd27908237a7abf2b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac00de34ee7c53c9f401ff4a12bd24fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 1, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a92de62224634376ba751b2437354a3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac00de34ee7c53c9f401ff4a12bd24fd
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.47696200013160706, 1.4755512475967407, 0.24671538174152374, 0.343254029750824], [1.2648696899414062, 1.5505101680755615, 0.3133050501346588, 3.9505667686462402]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_96e6a72de620c78d9cd8c3f8b8fb96d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13528534770011902], [0.12440189719200134], [0.05388212949037552], [0.32307666540145874], [0.08130541443824768]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.25348156690597534], [0.39336472749710083], [0.47650161385536194], [0.22819465398788452], [0.35213175415992737]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_29c93bec77e42ef2f6eb558d7c5e8a00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21230792999267578], [0.022249840199947357], [0.0162972342222929], [0.28616321086883545], [0.06640253961086273]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.47471386194229126], [0.35416045784950256], [0.40402311086654663], [0.4485560953617096], [0.46584224700927734]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_5aa2a9a85be280853b5798fe9a21d951(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18775731325149536], [0.12440189719200134], [0.05388212949037552], [0.37235766649246216], [0.08130541443824768]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.25348156690597534], [0.1419389396905899], [0.21522848308086395], [0.18140600621700287], [0.35213175415992737]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f919711b366ac28aec8c7693425b8b73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21230792999267578], [0.0743480697274208], [0.0162972342222929], [0.3133012354373932], [0.06640253961086273]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.47471386194229126], [0.33389967679977417], [0.40402311086654663], [0.4485560953617096], [0.46584224700927734]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_196981ba2a76d3677aeeafebfb6a8e1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13528534770011902], [0.3804410994052887], [0.13431361317634583], [0.32307666540145874], [0.13232417404651642]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1649070680141449], [0.39336472749710083], [0.47650161385536194], [0.22819465398788452], [0.16766460239887238]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f73ad6a1a155e1d7b097a3f8df362aa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4135350286960602], [0.022249840199947357], [0.3079543709754944], [0.28616321086883545], [0.11313261836767197]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21730045974254608], [0.35416045784950256], [0.32481035590171814], [0.011503858491778374], [0.4248878061771393]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_8ea99df7c5ff9f91f15971d744d499b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01143362931907177], [0.00884125754237175], [0.06832607835531235], [0.0002330932766199112], [0.1191963478922844]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_33dbe68b90409b8cf498039a3b935c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18775731325149536], [0.3804410994052887], [0.13431361317634583], [0.37235766649246216], [0.13232417404651642]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1649070680141449], [0.1419389396905899], [0.21522848308086395], [0.18140600621700287], [0.16766460239887238]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e19db7aaf238c237468cf7a778d7736e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4135350286960602], [0.0743480697274208], [0.3079543709754944], [0.3133012354373932], [0.11313261836767197]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21730045974254608], [0.33389967679977417], [0.32481035590171814], [0.011503858491778374], [0.4248878061771393]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_14766abafd2f4833b80e9c3738599889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.004484008066356182], [-0.06190362200140953], [0.001363899908028543], [0.05762871354818344], [0.01101756189018488]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.01143362931907177], [0.00884125754237175], [0.06832607835531235], [0.0002330933784833178], [0.1191963478922844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_4950506bd28b48f546858b689a053e85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-1.549868106842041], [1.1428229808807373], [-49.096107482910156], [0.995955228805542], [-9.818758964538574]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_d48b2fdd2314526f4a7d6d96038809d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13e483f6693cab66aa39f85c3d01bbda
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_968a0992d3909a6ee595362f606d54eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10b670757eab61b0e20c750f4be77f26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_373b469b29870b96c65f0c9c9e0d8a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_177bfe02fa886992bbd12d68c2c1e16d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_968a0992d3909a6ee595362f606d54eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_532a05ff9b90ec99fdd8c69ef7f95cd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72673d9b22b8b2dad3bdcb42685a4ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1773853541a1026734b909e97b57f9d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_299abd15ca851cb11b5072987927afc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_532a05ff9b90ec99fdd8c69ef7f95cd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4699a018b0b839403053ecf4a850148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f14bb32e712ab475e8d080a7918522f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b03e870dc05d049ae8e96b0b5681cc7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0fcf248a6e3ccbe3a9bc363fe8427bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4699a018b0b839403053ecf4a850148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49aca18bde6bb7c43a64981fd213e776(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13e483f6693cab66aa39f85c3d01bbda
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9693ea3b73c2082899dd418ae8a32da4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c667c9c3b487093ab1b393268f3418be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1602ebfcca274c35a2624a67410f7121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c667c9c3b487093ab1b393268f3418be
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26a72e6dcf59f364390962bc6c987194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.023670200258493423, 0.06822778284549713, 0.09304618835449219, 0.1754312664270401, 0.09071575105190277, 0.4499177932739258, 0.03357797861099243, 0.24033771455287933, 0.2845117151737213, 0.14653056859970093, 0.3833077549934387, 0.40311363339424133, 0.47563251852989197, 0.23145411908626556, 0.440609872341156, 0.4829781949520111, 0.3875403106212616, 0.4994688332080841, 0.3966350555419922, 0.34343641996383667], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_b2e2b6b2945f23aca9d8ce9cfee0582e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.023670200258493423, 0.06822778284549713, 0.09304618835449219, 0.1754312664270401, 0.09071575105190277, 0.4499177932739258, 0.03357797861099243, 0.24033771455287933, 0.2845117151737213, 0.14653056859970093, 0.3833077549934387, 0.40311363339424133, 0.47563251852989197, 0.23145411908626556, 0.440609872341156, 0.4829781949520111, 0.3875403106212616, 0.4994688332080841, 0.3966350555419922, 0.34343641996383667], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_c5eecbf8883c0772b7dbc216fd0e7ca5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1857021450996399], [0.232812762260437], [0.22833964228630066], [0.3062015175819397]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.43446049094200134], [0.3892061412334442], [0.21863189339637756], [0.3478715717792511]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_bf3866e5441a1785039326ea620e3887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28646060824394226], [0.2347613275051117], [0.07614689320325851], [0.03166484087705612]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28952881693840027], [0.428699254989624], [0.3750646412372589], [0.1684504747390747]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_1fd35fee353cf4c8b1c428c785a96fe8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1857021450996399], [0.232812762260437], [0.44483304023742676], [0.43192264437675476]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.43446049094200134], [0.3114164471626282], [0.21863189339637756], [0.22740739583969116]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7878a37f1bce8fe1b64816c8a272c524(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28646060824394226], [0.2347613275051117], [0.07614689320325851], [0.21912746131420135]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28952881693840027], [0.0999356284737587], [0.3750646412372589], [0.1684504747390747]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6e39a614a3ce971e31d0c747089b308c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.37367141246795654], [0.26348745822906494], [0.22833964228630066], [0.3062015175819397]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.05875740945339203], [0.3892061412334442], [0.20329052209854126], [0.3478715717792511]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a61d231880ec45c768dd15a99b1580dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3861541450023651], [0.4490680992603302], [0.4151049554347992], [0.03166484087705612]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2514371871948242], [0.428699254989624], [0.0340176485478878], [0.16604164242744446]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6b085233ef96143271331b41cb9658eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.043187495321035385], [-0.013158541172742844], [-0.05806963890790939], [0.015963705256581306]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_5f3fa339a683b5234e96831b6db647a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.37367141246795654], [0.26348745822906494], [0.44483304023742676], [0.43192264437675476]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.05875740945339203], [0.3114164471626282], [0.20329052209854126], [0.22740739583969116]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_00924913e0954cb67b21a149e9003730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3861541450023651], [0.4490680992603302], [0.4151049554347992], [0.21912746131420135]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2514371871948242], [0.0999356284737587], [0.0340176485478878], [0.16604164242744446]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_521e1f0e10f8d5ea5f48e735f34ac979(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04242425411939621], [-0.01673356629908085], [0.09204878658056259], [0.010856859385967255]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.043187495321035385], [-0.013158541172742844], [-0.05806963890790939], [0.015963705256581306]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_1ec6a5f66b6535174fa69a13e8b4d0ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.017990680411458015], [0.21364393830299377], [1.630857229232788], [-0.47037965059280396]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_eb1bcf2f4d9c64d512575f88861c1f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba82ccd26b68670f085e38f4f74ebad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_320a199560d8642d0673f59e74db6305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad1ecf87ccfeb2618d3ef1463541a4ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebf0eda5c97a28b86bf8fc08b38fafd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bba82ccd26b68670f085e38f4f74ebad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1602ebfcca274c35a2624a67410f7121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c667c9c3b487093ab1b393268f3418be
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4c1610691774867c5c6279ef720d84e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13e483f6693cab66aa39f85c3d01bbda
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_661343722c9ada9d2a980ea862ad4b5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6804, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_413556b6ad9fc50d926e0752773e0c8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_661343722c9ada9d2a980ea862ad4b5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f8d7e2242c40e9d273e19db2c29251f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4246392548084259, 0.0044077010825276375, 0.363427072763443, 0.25915592908859253], [0.24777847528457642, 0.29430466890335083, 0.4828647971153259, 0.15868110954761505], [0.0921330377459526, 0.49571457505226135, 0.3621473014354706, 0.3920249938964844], [0.0921330377459526, 0.49571457505226135, 0.3621473014354706, 0.3920249938964844], [0.13976053893566132, 0.31691572070121765, 0.0488075390458107, 0.3355359733104706]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.48885658383369446, 0.05222732201218605, 0.19859659671783447, 0.4256027340888977], [0.13974691927433014, 0.2562328279018402, 0.2397736757993698, 0.17280808091163635], [0.236659437417984, 0.04056683927774429, 0.24870304763317108, 0.4573097825050354], [0.236659437417984, 0.04056683927774429, 0.24870304763317108, 0.4573097825050354], [0.48069483041763306, 0.46220770478248596, 0.3727046847343445, 0.08313346654176712]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94faa81c16b9a2e23e9fed7cfc0f7620(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2273406391847ae89dcdc798c6512871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_100fbe70a24a5f2129a61350bbb4024b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d80a42d3ccd88f7d77138e53e5f9503
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b612ba4d37196e21d1ea3dff98a150a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501d18fa4962172203674bfa6aa0d6db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2273406391847ae89dcdc798c6512871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0810d908c6099ab2051b2bc9230d8ab
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4d29ddb3eafba200090c2cc96802dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49095484614372253, 0.05056574568152428, 0.22602437436580658, 0.2702874541282654], [0.2602322995662689, 0.2657991945743561, 0.1831132471561432, 0.1648610383272171], [0.09692883491516113, 0.0625140592455864, 0.11084946990013123, 0.36510151624679565], [0.49095484614372253, 0.05056574568152428, 0.22602437436580658, 0.2702874541282654], [0.24721501767635345, 0.11603589355945587, 0.3381746709346771, 0.10935646295547485], [0.36704662442207336, 0.14998702704906464, 0.11647056043148041, 0.4022963047027588], [0.24721501767635345, 0.11603589355945587, 0.3381746709346771, 0.10935646295547485]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.35526230931282043, 0.4342276155948639, 0.4757392704486847, 0.37623098492622375], [0.3971581757068634, 0.4530878961086273, 0.2965506911277771, 0.4920250177383423], [0.2712678611278534, 0.15222889184951782, 0.2440001368522644, 0.34964802861213684], [0.35526230931282043, 0.4342276155948639, 0.4757392704486847, 0.37623098492622375], [0.36358195543289185, 0.20468197762966156, 0.11512523144483566, 0.3920916020870209], [0.21806785464286804, 0.33629345893859863, 0.04788525402545929, 0.1571417599916458], [0.36358195543289185, 0.20468197762966156, 0.11512523144483566, 0.3920916020870209]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7172f67709eafe3ce667f8b3a302484d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee81927185378170745721caa17cd437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.05118778347969055]], [[0.10903573036193848]], [[0.4847700595855713]], [[0.2270936667919159]], [[0.4217580258846283]], [[0.2526901066303253]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.5237996578216553]], [[0.6988669633865356]], [[0.5810272693634033]], [[0.6514535546302795]], [[0.5936262011528015]], [[0.7933114767074585]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_df770788cc453d2632a17c13e6851446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.41506168246269226]], [[0.11528967320919037]], [[0.15182477235794067]], [[0.2979087233543396]], [[0.05990821495652199]], [[0.09339520335197449]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7646562457084656]], [[0.5594422817230225]], [[0.608974814414978]], [[0.7140263915061951]], [[0.7158714532852173]], [[0.7036947011947632]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_263baae981c6452c4fc38dfdeb3d1bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6945f1db5df5b2f8eb987ce6f1638184(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e382303435a63e749e3c1618f0d570e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6133dae93b7d7a9e251cfe215b215e67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cabd32ccc91f955e85c15f279d1c714a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.22274713218212128, 0.11812928318977356]], [[0.40234601497650146, 0.32135093212127686]], [[0.24842727184295654, 0.3543384373188019]], [[0.4109976589679718, 0.028806988149881363]], [[0.09972511976957321, 0.48537272214889526]], [[0.24782226979732513, 0.12431900948286057]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.09539089351892471, 0.014189804904162884]], [[0.4132586717605591, 0.26804450154304504]], [[0.2966609001159668, 0.13498319685459137]], [[0.1327727735042572, 0.43529269099235535]], [[0.22661817073822021, 0.15752831101417542]], [[0.050448279827833176, 0.11166179925203323]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_9b37f421d7cc7a6e84002376f8451168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.025420304387807846, 0.09367721527814865]], [[0.14023251831531525, 0.34991103410720825]], [[0.418991357088089, 0.374828040599823]], [[0.07487571239471436, 0.2823382019996643]], [[0.3401561975479126, 0.3036860525608063]], [[0.23089678585529327, 0.15044544637203217]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.09539089351892471, 0.014189804904162884]], [[0.4132586717605591, 0.26804450154304504]], [[0.2966609001159668, 0.13498319685459137]], [[0.1327727735042572, 0.43529269099235535]], [[0.22661817073822021, 0.15752831101417542]], [[0.050448279827833176, 0.11166179925203323]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_4ed0c7097d7789d364ba570332039650(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[[[0.2746506929397583, 0.3692845404148102]], [[0.2529679834842682, 0.3538026511669159]], [[0.1918400079011917, 0.46419987082481384]], [[0.4446069300174713, 0.48151710629463196]], [[0.22252815961837769, 0.22237402200698853]], [[0.07176125049591064, 0.38622698187828064]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17ea37bfbf76b52dc3a6ec3dffd93937(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.37460729479789734, 0.4844105839729309, 0.181888610124588, 0.44207683205604553, 0.02534153312444687, 0.3211316466331482, 0.009519957937300205, 0.40302348136901855, 0.22982287406921387, 0.10425077378749847, 0.3179247975349426, 0.40923941135406494, 0.2358681559562683, 0.14795830845832825, 0.06801445782184601, 0.4440118670463562], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_b7f5cc56867c835ed114d2cd9efcb717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37460729479789734, 0.4844105839729309, 0.181888610124588, 0.44207683205604553, 0.02534153312444687, 0.3211316466331482, 0.009519957937300205, 0.40302348136901855, 0.22982287406921387, 0.10425077378749847, 0.3179247975349426, 0.40923941135406494, 0.2358681559562683, 0.14795830845832825, 0.06801445782184601, 0.4440118670463562], dtype='float32').reshape([16]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2eb52740bdd7fd2c54a34a7cfb0f200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2eb52740bdd7fd2c54a34a7cfb0f200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d90374de55d664db7fbdea96cd545649(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b02f42b1db1ca1710e95443e70c4989(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e327c46ad117c3058ed2c177c5c38902(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af661c7571eac02708e4c609e38e64c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d71057689b12f3a5555e946fa88155c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_599b3dfefba3119bcb4ff25be431050b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e327c46ad117c3058ed2c177c5c38902(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f43ab2165b83b58c7ab2b262e252c0b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17805549502372742, 0.08694770932197571, 0.24425704777240753, 0.132246196269989], [0.13605213165283203, 0.3651181161403656, 0.3882414996623993, 0.2905106246471405], [0.41753944754600525, 0.4236792027950287, 0.3029243052005768, 0.4083331525325775], [0.44502976536750793, 0.0707036629319191, 0.4251031279563904, 0.20904578268527985], [0.28272974491119385, 0.3605213761329651, 0.46954378485679626, 0.444469153881073]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.12191290408372879, 0.35174989700317383, 0.10305600613355637, 0.01692168414592743], [0.1374139040708542, 0.3359614610671997, 0.18675795197486877, 0.011088688857853413], [0.31233879923820496, 0.19828705489635468, 0.09374342858791351, 0.2324703186750412], [0.09410864859819412, 0.44495889544487, 0.17497994005680084, 0.22089767456054688], [0.315926194190979, 0.4338832497596741, 0.2439688891172409, 0.04640224948525429]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a09a1928098121640c7dcfb7c74f779f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2353b992d4f1334cc422450b6a9cd76b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03461017832159996, 0.0004527518176473677, 0.36749422550201416, 0.32795700430870056], [0.3849736750125885, 0.032286059111356735, 0.061482444405555725, 0.3820096254348755], [0.102759949862957, 0.4908919334411621, 0.49278682470321655, 0.4886923134326935], [0.3849736750125885, 0.032286059111356735, 0.061482444405555725, 0.3820096254348755], [0.102759949862957, 0.4908919334411621, 0.49278682470321655, 0.4886923134326935]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.004080449230968952, 0.28355368971824646, 0.0013239141553640366, 0.3698040843009949], [0.45567235350608826, 0.2550033628940582, 0.02040109969675541, 0.4188835024833679], [0.48355358839035034, 0.49027279019355774, 0.13922299444675446, 0.3933846354484558], [0.45567235350608826, 0.2550033628940582, 0.02040109969675541, 0.4188835024833679], [0.48355358839035034, 0.49027279019355774, 0.13922299444675446, 0.3933846354484558]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93789134cdc7acce06a130729c021eb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13368675112724304], [0.3421638607978821], [0.3978815972805023], [0.15133216977119446], [0.22187696397304535], [0.2794041037559509], [0.01730276271700859], [0.08600067347288132], [0.0719427838921547]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.38795778155326843], [0.4817730784416199], [0.1863391399383545], [0.44741085171699524], [0.29538026452064514], [0.3727307915687561], [0.4028562605381012], [0.3053695261478424], [0.31240081787109375]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_2f6f5d7a3ec320339490b32f1c85cb5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16374197602272034], [0.18657687306404114], [0.13615421950817108], [0.26141688227653503], [0.006397350691258907], [0.38319242000579834], [0.3071695864200592], [0.004688146524131298], [0.23858243227005005]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.49473410844802856], [0.3323400914669037], [0.3997913599014282], [0.4925849139690399], [0.17630773782730103], [0.12873531877994537], [0.08800354599952698], [0.43823152780532837], [0.4259171187877655]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_bc1739b7884f8088f9636cfccb2f05bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13368675112724304], [0.39139324426651], [0.3978815972805023], [0.2982294261455536], [0.2754661738872528], [0.3869280219078064], [0.10337948054075241], [0.465537965297699], [0.4861213266849518]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14307871460914612], [0.19619788229465485], [0.1863391399383545], [0.2124066799879074], [0.035387687385082245], [0.1282057762145996], [0.4028562605381012], [0.007067443337291479], [0.31240081787109375]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_db56da587f2c027e1e7880096e1eeaa4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16374197602272034], [0.22206810116767883], [0.18589580059051514], [0.26141688227653503], [0.006397350691258907], [0.38319242000579834], [0.3071695864200592], [0.40571993589401245], [0.43421873450279236]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14178644120693207], [0.06213818117976189], [0.3997913599014282], [0.11639508605003357], [0.015070164576172829], [0.12873531877994537], [0.05166096240282059], [0.269934743642807], [0.09819310158491135]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_8930c125f87ae71f9c24f2a6c8b43ad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2243945300579071], [0.3421638607978821], [0.4742037057876587], [0.15133216977119446], [0.22187696397304535], [0.2794041037559509], [0.01730276271700859], [0.08600067347288132], [0.0719427838921547]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.38795778155326843], [0.4817730784416199], [0.017573527991771698], [0.44741085171699524], [0.29538026452064514], [0.3727307915687561], [0.023896757513284683], [0.3053695261478424], [0.20216022431850433]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_ad3cd92c1671398a40ab6e3d546b6c74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4538259506225586], [0.18657687306404114], [0.13615421950817108], [0.37509968876838684], [0.4772617220878601], [0.39100539684295654], [0.4677458703517914], [0.004688146524131298], [0.23858243227005005]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.49473410844802856], [0.3323400914669037], [0.2790012061595917], [0.4925849139690399], [0.17630773782730103], [0.10014522075653076], [0.08800354599952698], [0.43823152780532837], [0.4259171187877655]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_b9cd13bac2d0d51e05320a707c3ad954(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.006484865676611662], [0.05156746506690979], [-0.11047624051570892], [0.0472310408949852], [-0.024203266948461533], [0.03868870064616203], [-0.07902292162179947], [0.15735942125320435], [0.08276878297328949]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_81949986093347ad04de44bf5971891b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2243945300579071], [0.39139324426651], [0.4742037057876587], [0.2982294261455536], [0.2754661738872528], [0.3869280219078064], [0.10337948054075241], [0.465537965297699], [0.4861213266849518]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14307871460914612], [0.19619788229465485], [0.017573527991771698], [0.2124066799879074], [0.035387687385082245], [0.1282057762145996], [0.023896757513284683], [0.007067443337291479], [0.20216022431850433]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_c2d4f9c220c884c758de11a51ab21aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4538259506225586], [0.22206810116767883], [0.18589580059051514], [0.37509968876838684], [0.4772617220878601], [0.39100539684295654], [0.4677458703517914], [0.40571993589401245], [0.43421873450279236]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14178644120693207], [0.06213818117976189], [0.2790012061595917], [0.11639508605003357], [0.015070164576172829], [0.10014522075653076], [0.05166096240282059], [0.269934743642807], [0.09819310158491135]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_9ab8f63c2a091889b7eb6088fc1ffcdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.02537374570965767], [0.031217578798532486], [-0.04251473769545555], [0.022202739492058754], [0.11096224188804626], [0.0752519965171814], [0.03307155892252922], [0.06225350871682167], [0.09541821479797363]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.006484865676611662], [0.05156746506690979], [-0.11047624051570892], [0.0472310408949852], [-0.024203266948461533], [0.03868870064616203], [-0.07902292162179947], [0.15735942125320435], [0.08276878297328949]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_215345af8b85a7022f2700203001b9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [-0.0], [0.0], [-0.0], [0.0], [-0.0], [0.0], [0.0]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.7444261312484741], [-0.6518726348876953], [-1.5985398292541504], [-1.1272618770599365], [1.2181216478347778], [0.4858780801296234], [3.3894524574279785], [-1.5277197360992432], [0.13256831467151642]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_038caaf8caddf0ed66edf1b7a16c2165(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7216cb943e51a117d96edb4973fa4f72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.18063703179359436]], [[0.2793530225753784]], [[0.04766196757555008]], [[0.008773964829742908]], [[0.3327653408050537]], [[0.4748668670654297]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.6437686085700989]], [[0.66782146692276]], [[0.7871782183647156]], [[0.6366011500358582]], [[0.5327322483062744]], [[0.6649988293647766]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_065049d6aa2dcbe354a6cfa9b103e62f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.27055323123931885]], [[0.030705591663718224]], [[0.3505762219429016]], [[0.4922664165496826]], [[0.22030052542686462]], [[0.06326745450496674]]], dtype='float32').reshape([6, 1, 1]),
                paddle.to_tensor([[[0.7747390270233154]], [[0.5769653916358948]], [[0.551421046257019]], [[0.8195163011550903]], [[0.7298972010612488]], [[0.6213920712471008]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_44f89166f16877852ef10e96d998037f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ceec77ac6582d3dadc02f6072b2e847(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7da647ea406f9dae2a111d5895e5b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79d849c3be33c301145579fc57ad72db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa254b4c474f9e2e1154af56fb39774e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ceec77ac6582d3dadc02f6072b2e847(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_498f8b9df0aa81f47d26c72bd312f338(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.16312061250209808, 0.039097484201192856, 0.31288108229637146, 0.4763258099555969], [0.4967135787010193, 0.12254718691110611, 0.134210467338562, 0.22143881022930145], [0.33521342277526855, 0.3220093846321106, 0.48359137773513794, 0.09311709553003311], [0.4967135787010193, 0.12254718691110611, 0.134210467338562, 0.22143881022930145], [0.33521342277526855, 0.3220093846321106, 0.48359137773513794, 0.09311709553003311], [0.11703887581825256, 0.3053649067878723, 0.2134171426296234, 0.02529965713620186], [0.11703887581825256, 0.3053649067878723, 0.2134171426296234, 0.02529965713620186]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.1903732866048813, 0.4341961145401001, 0.336517870426178, 0.34890449047088623], [0.29464438557624817, 0.1553640365600586, 0.4579049348831177, 0.3192165493965149], [0.061085209250450134, 0.1616355925798416, 0.03644360974431038, 0.09199950844049454], [0.29464438557624817, 0.1553640365600586, 0.4579049348831177, 0.3192165493965149], [0.061085209250450134, 0.1616355925798416, 0.03644360974431038, 0.09199950844049454], [0.46255189180374146, 0.1523735076189041, 0.4603758454322815, 0.29759618639945984], [0.46255189180374146, 0.1523735076189041, 0.4603758454322815, 0.29759618639945984]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_d3d1358f2a2a08bd6e088e7a1eeb4499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3d1358f2a2a08bd6e088e7a1eeb4499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8f9019af5d1b9974919cc09d4a01956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7ad4ff1c512290975badd6950da16fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a157008b124ce5baf968e62389b177c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45227816700935364, 0.2502661347389221, 0.16833464801311493, 0.15319602191448212, 0.18911296129226685, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_223fa79905c437c3155e1b451cee46f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.27197179198265076, 0.4429328143596649, 0.17026685178279877, 0.08001313358545303, 0.23915867507457733, 0.05345135182142258], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0c4289316532007c34eab4af1235391c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.22168461978435516, 0.48501694202423096, 0.36503466963768005, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146], dtype='float32').reshape([6]),
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.22442752122879028, 0.4289640784263611, 0.301647424697876, 0.0385211743414402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_c40ae14ea19d06f1d9002ddc3aff5dbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3964419662952423, 0.30897533893585205, 0.40568625926971436, 0.10646888613700867, 0.4039801359176636, 0.40599721670150757], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3058600425720215, 0.14679817855358124, 0.3744381070137024, 0.0002531889476813376, 0.2294905185699463, 0.37590494751930237], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_d2d904ca7c2ab8a7207e6a85f2521c6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.22168461978435516, 0.3048173189163208, 0.36503466963768005, 0.15319602191448212, 0.32081952691078186, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.39980247616767883, 0.4289640784263611, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4ea3dfb939dd636f6f5420582e9d1e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.30897533893585205, 0.17026685178279877, 0.10646888613700867, 0.4039801359176636, 0.05345135182142258], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3236333429813385, 0.14679817855358124, 0.3744381070137024, 0.4712778925895691, 0.4077642858028412, 0.37590494751930237], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f0018c260eae78b633642ceb9cab0603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45227816700935364, 0.3048173189163208, 0.39980247616767883, 0.15319602191448212, 0.32081952691078186, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_275571931a658b2b5045df1fc385ee20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3236333429813385, 0.4429328143596649, 0.17026685178279877, 0.4712778925895691, 0.4077642858028412, 0.05345135182142258], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_20b79edfd5ffe489967c0d0123d4fba4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02505781129002571, 0.017090944573283195, 0.004393713548779488, -0.0008124950109049678, 0.017373912036418915, 0.0037726969458162785], dtype='float32').reshape([6]),
                paddle.to_tensor([-0.0, -0.0, 0.0, 0.0, -0.0, -0.0], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_380f33707b7ae5cd8f494a176a384d65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3431909680366516, 0.27754172682762146, 0.2840685546398163, 0.12728255987167358, 0.25496625900268555, 0.12390746176242828], dtype='float32').reshape([6]),
                paddle.to_tensor([0.360000342130661, 0.43232473731040955, 0.294731080532074, 0.42513933777809143, 0.3514323830604553, 0.09111270308494568], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_852829e2715d76210d3c8a97cd9c4053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.29780256748199463, 0.2295193076133728, 0.16009734570980072, 0.27564552426338196, 0.32346147298812866, 0.03871399909257889], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3511509895324707, 0.22788676619529724, 0.3900621831417084, 0.053361035883426666, 0.31673532724380493, 0.39095109701156616], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_51e7df6c353abe09be8897f41b27ada3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.45227816700935364, 0.48501694202423096, 0.39980247616767883, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146], dtype='float32').reshape([6]),
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.22442752122879028, 0.10136908292770386, 0.301647424697876, 0.0385211743414402], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4f36692060e295d42634bd2cb1932b5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3964419662952423, 0.4429328143596649, 0.40568625926971436, 0.4712778925895691, 0.4077642858028412, 0.40599721670150757], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3058600425720215, 0.01610579900443554, 0.14992783963680267, 0.0002531889476813376, 0.2294905185699463, 0.023976648226380348], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_bef46a1c9a587ae7c8377e34146bbb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.2543535232543945, 0.5762419700622559, 1.3521130084991455, -0.07189424335956573, 0.5185477137565613, 1.2921454906463623], dtype='float32').reshape([6]),
                paddle.to_tensor([-1.33828866481781, -0.12711717188358307, -1.4831517934799194, -0.13169337809085846, 0.6631419658660889, 0.6102384924888611], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4069913b2590f9feb1a57b95a29d40a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3581ba3d0bf00748680cc0d6cf4eff7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d71057689b12f3a5555e946fa88155c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_599b3dfefba3119bcb4ff25be431050b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4069913b2590f9feb1a57b95a29d40a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_405ebd37ce24644d82b920dd3ffbbada(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e74a419d8dcd91fc0129f1f725790e44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.18319472670555115, 0.21686814725399017, 0.4732165038585663, 0.03563392162322998, 0.26207268238067627, 0.130903959274292, 0.44992533326148987, 0.0970231369137764, 0.4192184805870056, 0.3830379247665405, 0.41160717606544495, 0.21541158854961395, 0.13209235668182373, 0.42042574286460876, 0.40241438150405884, 0.11916881799697876, 0.17227095365524292, 0.34461653232574463, 0.4960813820362091, 0.1515713632106781, 0.1289307177066803, 0.3049981892108917, 0.004009498283267021, 0.4341105818748474], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_0119d8a5aad78c47aba505cc0d8f954e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.18319472670555115, 0.21686814725399017, 0.4732165038585663, 0.03563392162322998, 0.26207268238067627, 0.130903959274292, 0.44992533326148987, 0.0970231369137764, 0.4192184805870056, 0.3830379247665405, 0.41160717606544495, 0.21541158854961395, 0.13209235668182373, 0.42042574286460876, 0.40241438150405884, 0.11916881799697876, 0.17227095365524292, 0.34461653232574463, 0.4960813820362091, 0.1515713632106781, 0.1289307177066803, 0.3049981892108917, 0.004009498283267021, 0.4341105818748474], dtype='float32').reshape([24]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c881a936c25c8194f4bed0bc1fe53879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04d0fb7a1fc2b12fe24823aafa31e51e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_946c5fe09751cdab1ddf1303a837e4f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c777deba099f50da16fe63ba79b1a42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_945bb918603acc2712b4367c94692119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5015f5025e0809d598f06568bf7776b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_946c5fe09751cdab1ddf1303a837e4f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf43246186b966ab4007843fcd937f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a6792c661b11940d4bbd9a83d82bdea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([4]),
                paddle.to_tensor([0.05085877701640129, 0.22930851578712463, 0.13148881494998932, 0.16494080424308777], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_1124143059ea244197c4b0f83ac163cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.05085877701640129, 0.22930851578712463, 0.13148881494998932, 0.16494080424308777], dtype='float32').reshape([4]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([4]),
            ]


    
    class PrimitiveOp_1942fd0f5f0cba96318357b3f64ea82c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return input_0 - input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5bc9567b90cd870c3c14ac368ac8092(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1942fd0f5f0cba96318357b3f64ea82c
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_98a17ae805d7b9623b126839fbcc8fa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1942fd0f5f0cba96318357b3f64ea82c
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a337a9a6beb1ce29322f96287cf591b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0094315bc00efd8603c112614d3ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.025187794119119644, 0.23439233005046844, 0.3178129196166992, 0.09868212789297104], [0.0008286791271530092, 0.3812451660633087, 0.04935451224446297, 0.14829841256141663], [0.33462032675743103, 0.184892475605011, 0.21668128669261932, 0.3788776993751526], [0.16523830592632294, 0.30108246207237244, 0.4440777599811554, 0.17766731977462769], [0.16523830592632294, 0.30108246207237244, 0.4440777599811554, 0.17766731977462769], [0.33462032675743103, 0.184892475605011, 0.21668128669261932, 0.3788776993751526]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.29314377903938293, 0.13475337624549866, 0.26757317781448364, 0.0016862310003489256], [0.49155473709106445, 0.4800690710544586, 0.006142769008874893, 0.016341568902134895], [0.11181717365980148, 0.44043388962745667, 0.00794908031821251, 0.2214917689561844], [0.08348874747753143, 0.09139897674322128, 0.14279025793075562, 0.19967620074748993], [0.08348874747753143, 0.09139897674322128, 0.14279025793075562, 0.19967620074748993], [0.11181717365980148, 0.44043388962745667, 0.00794908031821251, 0.2214917689561844]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_d13659647a3ff52ade94368a0445de32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.42409706115722656, 0.487624853849411, 0.3005632758140564, 0.3299276530742645], [0.05977506563067436, 0.10115164518356323, 0.1450292468070984, 0.07699044793844223], [0.2550349235534668, 0.18359854817390442, 0.0963495597243309, 0.19211435317993164], [0.4906561076641083, 0.45334500074386597, 0.23444624245166779, 0.07382465898990631], [0.42409706115722656, 0.487624853849411, 0.3005632758140564, 0.3299276530742645]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.0006221520598046482, 0.03902745991945267, 0.32607606053352356, 0.49414271116256714], [0.4453412890434265, 0.42279648780822754, 0.0333394780755043, 0.0391206294298172], [0.2535695433616638, 0.045391522347927094, 0.02295094169676304, 0.22988493740558624], [0.1734488159418106, 0.3022133409976959, 0.45550966262817383, 0.16303986310958862], [0.0006221520598046482, 0.03902745991945267, 0.32607606053352356, 0.49414271116256714]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_7e6ac12dce03e35fbf8971aa6b9b3536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae908fea3801793f51f3c10a6d0241f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27537572383880615]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.20313051342964172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_13c8400d047220d0856726d7be87927b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14600974321365356]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.15484283864498138]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_82a82f38da0268770e1bbc84f9679476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.27537572383880615]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.178619846701622]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_d370bca47285bdaf216ae2bafbd8d803(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.19937725365161896]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.15484283864498138]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_e216a54cba76f00a6b4d149cdee67861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31011876463890076]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.20313051342964172]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_be06af25cbc0542d9c38c46e92090569(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14600974321365356]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.004780620336532593]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_1ed01eb0bdd00df7b251bcf9491c42b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01941882260143757]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_f0af44b2d9fbc800b988476f0b8c8e8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.31011876463890076]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.178619846701622]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_01071e90ddb102542b43fbd437c53988(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.19937725365161896]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.004780620336532593]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_0b2e20a998479a0a7d4e58352c26cc56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.025589246302843094]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.01941882260143757]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_75d97d106cb27ed001c1a4c23fc11d07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.24113346636295319]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_6557fae0c6fc0a9827866d0538cb2208(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0779203251004219], [0.256033331155777], [0.2259289026260376], [0.08137910068035126], [0.0947040542960167], [0.32221317291259766]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3637081980705261], [0.39640137553215027], [0.31833428144454956], [0.4412391483783722], [0.43500569462776184], [0.4164228141307831]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_bc5c6047c9d16af3e712e0ddc148a1ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4113561809062958], [0.32371410727500916], [0.34662380814552307], [0.24301113188266754], [0.09847179055213928], [0.15297983586788177]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.2915404736995697], [0.1776416003704071], [0.19010786712169647], [0.35384827852249146], [0.26003992557525635], [0.24949562549591064]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_5b344c820ae622df6c36df9500459e82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28181594610214233], [0.256033331155777], [0.3946605324745178], [0.4794350862503052], [0.0947040542960167], [0.32221317291259766]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.3637081980705261], [0.39640137553215027], [0.31833428144454956], [0.4412391483783722], [0.11331885308027267], [0.4164228141307831]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_c3ffdbc6df3359afdc8714ec2cf4007a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4542856514453888], [0.32371410727500916], [0.47006481885910034], [0.24301113188266754], [0.31819093227386475], [0.3337242603302002]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.2915404736995697], [0.06591049581766129], [0.19010786712169647], [0.17498598992824554], [0.26003992557525635], [0.24949562549591064]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_60467d731a1556caef9060895a9da9cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0779203251004219], [0.28568974137306213], [0.2259289026260376], [0.08137910068035126], [0.37925606966018677], [0.45194125175476074]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.269661545753479], [0.24576835334300995], [0.09308407455682755], [0.3879702687263489], [0.43500569462776184], [0.14424768090248108]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_1d23722bd9948552ba27bd26b4ca1f28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4113561809062958], [0.38559842109680176], [0.34662380814552307], [0.25977855920791626], [0.09847179055213928], [0.15297983586788177]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.25507786870002747], [0.1776416003704071], [0.0561361089348793], [0.35384827852249146], [0.1015116423368454], [0.23805175721645355]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_7b699185b21847a647bc7ed68fdc0822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.043292563408613205], [-0.02788546308875084], [0.05995785444974899], [0.03143922984600067], [-0.0009129986283369362], [-0.034111231565475464]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8542b1dc1eafd7abffe407ac22e9400b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28181594610214233], [0.28568974137306213], [0.3946605324745178], [0.4794350862503052], [0.37925606966018677], [0.45194125175476074]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.269661545753479], [0.24576835334300995], [0.09308407455682755], [0.3879702687263489], [0.11331885308027267], [0.14424768090248108]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_52a4e2248f968d31941d20bbaaa27d15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4542856514453888], [0.38559842109680176], [0.47006481885910034], [0.25977855920791626], [0.31819093227386475], [0.3337242603302002]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.25507786870002747], [0.06591049581766129], [0.0561361089348793], [0.17498598992824554], [0.1015116423368454], [0.23805175721645355]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_dc9e8005470dd0488630b82991647e27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.002421251032501459], [0.012762386351823807], [0.12483116239309311], [0.007755537051707506], [0.057623084634542465], [0.02943781390786171]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[-0.043292563408613205], [-0.02788546308875084], [0.05995785444974899], [0.03143922984600067], [-0.0009129985119216144], [-0.034111231565475464]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_a23cf7642168fe1a0e9be0478771cb53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0], [-0.0], [0.0], [0.0], [-0.0], [-0.0]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[18.880247116088867], [3.1849725246429443], [0.5196884274482727], [-3.053778648376465], [1.0158443450927734], [2.1587555408477783]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_8938a22bdbc164cbf0aca9fd52ca546e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2748917043209076, 0.08581192046403885, 0.4370581805706024, 0.22711123526096344], [0.2532287836074829, 0.10648109763860703, 0.32183951139450073, 0.3253396153450012], [0.24018266797065735, 0.39429283142089844, 0.3479117155075073, 0.4558558762073517], [0.025166206061840057, 0.25497257709503174, 0.1559371054172516, 0.08968935906887054]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([[0.30902770161628723, 0.3745928108692169, 0.090107761323452, 0.03648531064391136], [0.023343751206994057, 0.16288305819034576, 0.36057403683662415, 0.0980585440993309], [0.2143491953611374, 0.08171876519918442, 0.4514274299144745, 0.21363447606563568], [0.3628651201725006, 0.32337111234664917, 0.040081609040498734, 0.15516340732574463]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_173101cff0edc06a837528547f1b8fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a47c291ab79d953081b8f1f3a13f3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a75323c774ff55e2853da30d2527d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b84edc75e5f3ef32a2f874087ae7ca04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b53b58d42b9a73bd1ea0c64f0412ffaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad1ecf87ccfeb2618d3ef1463541a4ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebf0eda5c97a28b86bf8fc08b38fafd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b84edc75e5f3ef32a2f874087ae7ca04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac684e8252a13736150650883a246c0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2859324514865875, 0.40432310104370117, 0.14047105610370636, 0.1841762661933899], [0.2859324514865875, 0.40432310104370117, 0.14047105610370636, 0.1841762661933899], [0.33153173327445984, 0.33179908990859985, 0.12240845710039139, 0.06773053109645844], [0.3055313229560852, 0.4260094463825226, 0.10106411576271057, 0.32232150435447693], [0.17913493514060974, 0.2465459108352661, 0.1178114265203476, 0.0793425589799881], [0.004713766276836395, 0.02173936739563942, 0.19002436101436615, 0.2897177040576935], [0.009822460822761059, 0.3363867402076721, 0.18706294894218445, 0.48170849680900574]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.14499889314174652, 0.40469223260879517, 0.4757552444934845, 0.10206005722284317], [0.14499889314174652, 0.40469223260879517, 0.4757552444934845, 0.10206005722284317], [0.44132161140441895, 0.005613656714558601, 0.0779615044593811, 0.04862334579229355], [0.09588927030563354, 0.43522098660469055, 0.37613382935523987, 0.46532732248306274], [0.24823467433452606, 0.010199218988418579, 0.3539154529571533, 0.3569227159023285], [0.30644461512565613, 0.336038738489151, 0.31734803318977356, 0.3516157567501068], [0.10263757407665253, 0.46752721071243286, 0.16771270334720612, 0.4427560567855835]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bf17f7c12766d8b1a40f5233d27dfb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_babac39d4ce4938641a9d54fb45d9121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27e532bf84898ad21ca458d5fee51bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a5568eaeb925524dbbf55539da0280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2211645ab1b11819ca1eae253624ea5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0ddc0eaf16bb43fd2ad7a3644c7757b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e27e532bf84898ad21ca458d5fee51bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2abd0bd1b7cdad8f899f07117021eefa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_536d30d63ef05388848b92ffee2ed8be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c3f7c50469205aa8991ff5a4cd21352(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38ea7f0e6fc4386f37a8ceecbe4459dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2abd0bd1b7cdad8f899f07117021eefa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9633b487c45d7cf36528b464d78896bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f0923bdb1c1ee9dcbeb6d6085f08a47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25175294280052185, 0.39583370089530945, 0.476584792137146, 0.11600856482982635], [0.19268378615379333, 0.33271491527557373, 0.48918667435646057, 0.00018414220539852977], [0.19268378615379333, 0.33271491527557373, 0.48918667435646057, 0.00018414220539852977], [0.42016997933387756, 0.4741819500923157, 0.3954072594642639, 0.020229332149028778], [0.31349611282348633, 0.2956351637840271, 0.3755139708518982, 0.13446329534053802], [0.004371978342533112, 0.30586355924606323, 0.08176377415657043, 0.4264273941516876]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([[0.1815273016691208, 0.15519441664218903, 0.2377370148897171, 0.21099913120269775], [0.15358561277389526, 0.48909053206443787, 0.28568097949028015, 0.2635892927646637], [0.15358561277389526, 0.48909053206443787, 0.28568097949028015, 0.2635892927646637], [0.488042950630188, 0.4658406674861908, 0.28834664821624756, 0.2730041742324829], [0.018005892634391785, 0.48755109310150146, 0.45424145460128784, 0.039086345583200455], [0.11298181861639023, 0.40279942750930786, 0.24674753844738007, 0.23806215822696686]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25573400350c2b90a63e00e4a08da768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c727b9dd7ef7f03d1b90627575b81b22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddf1c370d2d9ca55cb9a828d1835245e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([100, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.5930296182632446, 5.424015045166016, 0.8112000823020935, 0.36732006072998047], [0.827667236328125, 0.2817864716053009, 0.6280726790428162, 1.0955476760864258]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f4a60ab537e4f357c136088b519e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_622bfedbba21eca26db2e0840b31c685(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abe3bfd3b84c2bd27908237a7abf2b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48e9cac810fa6c12677e73f8a7ea9366(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([300, 1, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.47696200013160706, 1.4755512475967407, 0.24671538174152374, 0.343254029750824], [1.2648696899414062, 1.5505101680755615, 0.3133050501346588, 3.9505667686462402]], dtype='float32').reshape([2, 4]),
            ]


    class TestPrimitiveOp_96e6a72de620c78d9cd8c3f8b8fb96d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13528534770011902], [0.12440189719200134], [0.05388212949037552], [0.32307666540145874], [0.08130541443824768]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.25348156690597534], [0.39336472749710083], [0.47650161385536194], [0.22819465398788452], [0.35213175415992737]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_29c93bec77e42ef2f6eb558d7c5e8a00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21230792999267578], [0.022249840199947357], [0.0162972342222929], [0.28616321086883545], [0.06640253961086273]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.47471386194229126], [0.35416045784950256], [0.40402311086654663], [0.4485560953617096], [0.46584224700927734]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_5aa2a9a85be280853b5798fe9a21d951(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18775731325149536], [0.12440189719200134], [0.05388212949037552], [0.37235766649246216], [0.08130541443824768]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.25348156690597534], [0.1419389396905899], [0.21522848308086395], [0.18140600621700287], [0.35213175415992737]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f919711b366ac28aec8c7693425b8b73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21230792999267578], [0.0743480697274208], [0.0162972342222929], [0.3133012354373932], [0.06640253961086273]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.47471386194229126], [0.33389967679977417], [0.40402311086654663], [0.4485560953617096], [0.46584224700927734]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_196981ba2a76d3677aeeafebfb6a8e1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13528534770011902], [0.3804410994052887], [0.13431361317634583], [0.32307666540145874], [0.13232417404651642]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1649070680141449], [0.39336472749710083], [0.47650161385536194], [0.22819465398788452], [0.16766460239887238]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f73ad6a1a155e1d7b097a3f8df362aa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4135350286960602], [0.022249840199947357], [0.3079543709754944], [0.28616321086883545], [0.11313261836767197]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21730045974254608], [0.35416045784950256], [0.32481035590171814], [0.011503858491778374], [0.4248878061771393]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_8ea99df7c5ff9f91f15971d744d499b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.01143362931907177], [0.00884125754237175], [0.06832607835531235], [0.0002330932766199112], [0.1191963478922844]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_33dbe68b90409b8cf498039a3b935c36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.18775731325149536], [0.3804410994052887], [0.13431361317634583], [0.37235766649246216], [0.13232417404651642]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.1649070680141449], [0.1419389396905899], [0.21522848308086395], [0.18140600621700287], [0.16766460239887238]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e19db7aaf238c237468cf7a778d7736e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4135350286960602], [0.0743480697274208], [0.3079543709754944], [0.3133012354373932], [0.11313261836767197]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21730045974254608], [0.33389967679977417], [0.32481035590171814], [0.011503858491778374], [0.4248878061771393]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_14766abafd2f4833b80e9c3738599889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.004484008066356182], [-0.06190362200140953], [0.001363899908028543], [0.05762871354818344], [0.01101756189018488]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.01143362931907177], [0.00884125754237175], [0.06832607835531235], [0.0002330933784833178], [0.1191963478922844]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_4950506bd28b48f546858b689a053e85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[-1.549868106842041], [1.1428229808807373], [-49.096107482910156], [0.995955228805542], [-9.818758964538574]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_f84af6559da51173c9d19bdbd6194650(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1133f1eb5ac99d918e6479c9cb16ab69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06c8e7cf2e848648d9572b0380beef16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf707174e27be296818891bb25e5e8c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd286658adc288df238c0e07bc852ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_373b469b29870b96c65f0c9c9e0d8a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_177bfe02fa886992bbd12d68c2c1e16d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf707174e27be296818891bb25e5e8c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da8f4950b674056da51cb4d9c1f99f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d490363fd3d8da4c4cc420522ef026c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1773853541a1026734b909e97b57f9d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_299abd15ca851cb11b5072987927afc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da8f4950b674056da51cb4d9c1f99f65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ca02f40a415c394a85d654de93b40e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48009f08251d5528ed911b1f855826f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b03e870dc05d049ae8e96b0b5681cc7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0fcf248a6e3ccbe3a9bc363fe8427bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ca02f40a415c394a85d654de93b40e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_281517e5a461f2b1fcc608f9ea7d45a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9693ea3b73c2082899dd418ae8a32da4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0028e2987fdbaccf62e268ba1a793a33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3520bcaf585eef8339d83a809b5e6af2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26a72e6dcf59f364390962bc6c987194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.023670200258493423, 0.06822778284549713, 0.09304618835449219, 0.1754312664270401, 0.09071575105190277, 0.4499177932739258, 0.03357797861099243, 0.24033771455287933, 0.2845117151737213, 0.14653056859970093, 0.3833077549934387, 0.40311363339424133, 0.47563251852989197, 0.23145411908626556, 0.440609872341156, 0.4829781949520111, 0.3875403106212616, 0.4994688332080841, 0.3966350555419922, 0.34343641996383667], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_b2e2b6b2945f23aca9d8ce9cfee0582e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97bb131ea97a8cf0466245328fb49165
        def get_inputs(self):
            return [
                paddle.to_tensor([0.023670200258493423, 0.06822778284549713, 0.09304618835449219, 0.1754312664270401, 0.09071575105190277, 0.4499177932739258, 0.03357797861099243, 0.24033771455287933, 0.2845117151737213, 0.14653056859970093, 0.3833077549934387, 0.40311363339424133, 0.47563251852989197, 0.23145411908626556, 0.440609872341156, 0.4829781949520111, 0.3875403106212616, 0.4994688332080841, 0.3966350555419922, 0.34343641996383667], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_c5eecbf8883c0772b7dbc216fd0e7ca5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1857021450996399], [0.232812762260437], [0.22833964228630066], [0.3062015175819397]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.43446049094200134], [0.3892061412334442], [0.21863189339637756], [0.3478715717792511]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_bf3866e5441a1785039326ea620e3887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28646060824394226], [0.2347613275051117], [0.07614689320325851], [0.03166484087705612]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28952881693840027], [0.428699254989624], [0.3750646412372589], [0.1684504747390747]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_1fd35fee353cf4c8b1c428c785a96fe8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1857021450996399], [0.232812762260437], [0.44483304023742676], [0.43192264437675476]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.43446049094200134], [0.3114164471626282], [0.21863189339637756], [0.22740739583969116]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_7878a37f1bce8fe1b64816c8a272c524(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.28646060824394226], [0.2347613275051117], [0.07614689320325851], [0.21912746131420135]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28952881693840027], [0.0999356284737587], [0.3750646412372589], [0.1684504747390747]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6e39a614a3ce971e31d0c747089b308c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.37367141246795654], [0.26348745822906494], [0.22833964228630066], [0.3062015175819397]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.05875740945339203], [0.3892061412334442], [0.20329052209854126], [0.3478715717792511]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a61d231880ec45c768dd15a99b1580dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3861541450023651], [0.4490680992603302], [0.4151049554347992], [0.03166484087705612]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2514371871948242], [0.428699254989624], [0.0340176485478878], [0.16604164242744446]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_6b085233ef96143271331b41cb9658eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.043187495321035385], [-0.013158541172742844], [-0.05806963890790939], [0.015963705256581306]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.0], [0.0], [0.0], [0.0]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_5f3fa339a683b5234e96831b6db647a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.37367141246795654], [0.26348745822906494], [0.44483304023742676], [0.43192264437675476]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.05875740945339203], [0.3114164471626282], [0.20329052209854126], [0.22740739583969116]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_00924913e0954cb67b21a149e9003730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3861541450023651], [0.4490680992603302], [0.4151049554347992], [0.21912746131420135]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2514371871948242], [0.0999356284737587], [0.0340176485478878], [0.16604164242744446]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_521e1f0e10f8d5ea5f48e735f34ac979(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04242425411939621], [-0.01673356629908085], [0.09204878658056259], [0.010856859385967255]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.043187495321035385], [-0.013158541172742844], [-0.05806963890790939], [0.015963705256581306]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_1ec6a5f66b6535174fa69a13e8b4d0ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0], [-0.0], [-0.0], [0.0]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[-0.017990680411458015], [0.21364393830299377], [1.630857229232788], [-0.47037965059280396]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_eb1bcf2f4d9c64d512575f88861c1f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f2c9c47d52aaa77aa61663ce9ef5088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53b7b6d680cba966a01815d55d699707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad1ecf87ccfeb2618d3ef1463541a4ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebf0eda5c97a28b86bf8fc08b38fafd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f2c9c47d52aaa77aa61663ce9ef5088(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3520bcaf585eef8339d83a809b5e6af2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15d04b4fe1c6fbcb2b12c53924665429(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b91d1605bb3091d391ea03d6f8746bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c54ef6b5eb58f7c46947adfe4594162(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0a189c700f923c014e513b319e236e8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f8d7e2242c40e9d273e19db2c29251f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4246392548084259, 0.0044077010825276375, 0.363427072763443, 0.25915592908859253], [0.24777847528457642, 0.29430466890335083, 0.4828647971153259, 0.15868110954761505], [0.0921330377459526, 0.49571457505226135, 0.3621473014354706, 0.3920249938964844], [0.0921330377459526, 0.49571457505226135, 0.3621473014354706, 0.3920249938964844], [0.13976053893566132, 0.31691572070121765, 0.0488075390458107, 0.3355359733104706]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([[0.48885658383369446, 0.05222732201218605, 0.19859659671783447, 0.4256027340888977], [0.13974691927433014, 0.2562328279018402, 0.2397736757993698, 0.17280808091163635], [0.236659437417984, 0.04056683927774429, 0.24870304763317108, 0.4573097825050354], [0.236659437417984, 0.04056683927774429, 0.24870304763317108, 0.4573097825050354], [0.48069483041763306, 0.46220770478248596, 0.3727046847343445, 0.08313346654176712]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93d41470e391e90aa6c61df1f78c23c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ee6567b417936367b7a224a6fa57550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94faa81c16b9a2e23e9fed7cfc0f7620(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcde006a3a48e634c9c568003fc5a104(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82bb5ac40cf65fed0d5c3f97139d5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b612ba4d37196e21d1ea3dff98a150a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a0fd9ef47d80ed7b7d5ac420b270a5c
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501d18fa4962172203674bfa6aa0d6db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da31d8326deea42fd6ce19c10eae97b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcde006a3a48e634c9c568003fc5a104(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4d29ddb3eafba200090c2cc96802dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49095484614372253, 0.05056574568152428, 0.22602437436580658, 0.2702874541282654], [0.2602322995662689, 0.2657991945743561, 0.1831132471561432, 0.1648610383272171], [0.09692883491516113, 0.0625140592455864, 0.11084946990013123, 0.36510151624679565], [0.49095484614372253, 0.05056574568152428, 0.22602437436580658, 0.2702874541282654], [0.24721501767635345, 0.11603589355945587, 0.3381746709346771, 0.10935646295547485], [0.36704662442207336, 0.14998702704906464, 0.11647056043148041, 0.4022963047027588], [0.24721501767635345, 0.11603589355945587, 0.3381746709346771, 0.10935646295547485]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([[0.35526230931282043, 0.4342276155948639, 0.4757392704486847, 0.37623098492622375], [0.3971581757068634, 0.4530878961086273, 0.2965506911277771, 0.4920250177383423], [0.2712678611278534, 0.15222889184951782, 0.2440001368522644, 0.34964802861213684], [0.35526230931282043, 0.4342276155948639, 0.4757392704486847, 0.37623098492622375], [0.36358195543289185, 0.20468197762966156, 0.11512523144483566, 0.3920916020870209], [0.21806785464286804, 0.33629345893859863, 0.04788525402545929, 0.1571417599916458], [0.36358195543289185, 0.20468197762966156, 0.11512523144483566, 0.3920916020870209]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f1bb5ddef0382ed0a7a257e1af7bd72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86702e1b01684d81107fc9d2e782dd77
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7172f67709eafe3ce667f8b3a302484d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6084071aa209c01df02838230696ee7c
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()