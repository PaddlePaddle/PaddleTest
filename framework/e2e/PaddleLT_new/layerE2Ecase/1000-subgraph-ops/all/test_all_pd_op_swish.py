import os
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
    class PrimitiveOp_bfc02a2c33b8f367d2cdbbee1a98ee6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11479d90a7178936e56a8de59e6ddb6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfc02a2c33b8f367d2cdbbee1a98ee6b
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc01d7777bcc60aebfff17850f1d5e74(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c8bbcce91412cfa618e97f9896024fed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc01d7777bcc60aebfff17850f1d5e74
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6ad495e2146aa307d8d30704fbd2547a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_742c118f8f70544da31b4e7578469b7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ad495e2146aa307d8d30704fbd2547a
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0abbffceca9d2af523b60f83df6a307c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bbade16ed4ba72ef3c167d3b586adf32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0abbffceca9d2af523b60f83df6a307c
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11479d90a7178936e56a8de59e6ddb6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfc02a2c33b8f367d2cdbbee1a98ee6b
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f23253ee1f7a92b35a475d2fb2413da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc01d7777bcc60aebfff17850f1d5e74
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_742c118f8f70544da31b4e7578469b7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ad495e2146aa307d8d30704fbd2547a
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86ae2d67502ffd297c104c69b66347c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1152, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98f194be72af0bd5429c4ce0ebe22bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86ae2d67502ffd297c104c69b66347c6
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_162ed8c85f2ae4b86307b7f30eb4c7a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90507e31c569dad1f95ed3f5705da21c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_162ed8c85f2ae4b86307b7f30eb4c7a4
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54da6b5dbf10ab3028f20c9628ea747f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0abbffceca9d2af523b60f83df6a307c
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11479d90a7178936e56a8de59e6ddb6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfc02a2c33b8f367d2cdbbee1a98ee6b
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c400c160e5834c191afc32e293822d72(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6d1d867cf43fa7a5aa16b36e7c9c0db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c400c160e5834c191afc32e293822d72
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_72ad31459f24d8927b948415081258b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99cefb534aa11ebc4a0afda45d8656a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72ad31459f24d8927b948415081258b2
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_432689871283dd916827a53ff507e4b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c38c484a14b48adb0db8ede583687d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_432689871283dd916827a53ff507e4b1
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e99b9e2d091b4f081f81111ec3b8425d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9fc4146703ab3ec99f7846905a65079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e99b9e2d091b4f081f81111ec3b8425d
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6858da4ccd9e87df4db75c152680e087(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f199ade29449b4bf203f8a4044458e42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6858da4ccd9e87df4db75c152680e087
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_79d492005e5ad10afc7b231674cda7e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d90b729e9c779d35f125a000aa70654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79d492005e5ad10afc7b231674cda7e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f23253ee1f7a92b35a475d2fb2413da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc01d7777bcc60aebfff17850f1d5e74
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_742c118f8f70544da31b4e7578469b7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ad495e2146aa307d8d30704fbd2547a
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28ccca0c62ab5d0b22b49c403ed1ded5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6858da4ccd9e87df4db75c152680e087
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c68953c905382180fe1e3b0e30973631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79d492005e5ad10afc7b231674cda7e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f697cec2a464c6998b8575a49940dd9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0abbffceca9d2af523b60f83df6a307c
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a2eef232883489590b2f677c93370f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfc02a2c33b8f367d2cdbbee1a98ee6b
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c38c484a14b48adb0db8ede583687d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_432689871283dd916827a53ff507e4b1
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9fc4146703ab3ec99f7846905a65079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e99b9e2d091b4f081f81111ec3b8425d
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0126ca145c77be5e9475086efbd083d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_162ed8c85f2ae4b86307b7f30eb4c7a4
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e677e14b383b08c07760cb800dfa1caa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_23579de53db0e3e6f461085bd5740c9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e677e14b383b08c07760cb800dfa1caa
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e4bb927405d471c3245c40599e864ae0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0207d017e21bd96447146d956a5c4327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4bb927405d471c3245c40599e864ae0
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_225e9564c413515419830805f4ec73b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_432689871283dd916827a53ff507e4b1
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c87cd83fc90c41d48338ef37e3de1998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e99b9e2d091b4f081f81111ec3b8425d
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f697cec2a464c6998b8575a49940dd9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0abbffceca9d2af523b60f83df6a307c
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a2eef232883489590b2f677c93370f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfc02a2c33b8f367d2cdbbee1a98ee6b
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c9697c083bce1b6cdffc6088620c4a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6858da4ccd9e87df4db75c152680e087
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c68953c905382180fe1e3b0e30973631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79d492005e5ad10afc7b231674cda7e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5f161c596ae007d841cdd9ada7ed317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ad495e2146aa307d8d30704fbd2547a
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cc55219be8fc87f9587ee6d306b9ee0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e677e14b383b08c07760cb800dfa1caa
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62d130bc5cdb425ebda3eba70c32fbc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4bb927405d471c3245c40599e864ae0
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_578b9cb8de1f6f0ff86954e5a0d541d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6858da4ccd9e87df4db75c152680e087
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d90b729e9c779d35f125a000aa70654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79d492005e5ad10afc7b231674cda7e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d90b729e9c779d35f125a000aa70654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79d492005e5ad10afc7b231674cda7e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_786d6123c9d4f1d93d36bb9fd4c49153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86ae2d67502ffd297c104c69b66347c6
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0126ca145c77be5e9475086efbd083d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_162ed8c85f2ae4b86307b7f30eb4c7a4
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c9697c083bce1b6cdffc6088620c4a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6858da4ccd9e87df4db75c152680e087
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c68953c905382180fe1e3b0e30973631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79d492005e5ad10afc7b231674cda7e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11479d90a7178936e56a8de59e6ddb6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfc02a2c33b8f367d2cdbbee1a98ee6b
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8bbcce91412cfa618e97f9896024fed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc01d7777bcc60aebfff17850f1d5e74
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_742c118f8f70544da31b4e7578469b7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ad495e2146aa307d8d30704fbd2547a
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5f161c596ae007d841cdd9ada7ed317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ad495e2146aa307d8d30704fbd2547a
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea06924d92425333dd0ad7583cc84557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c400c160e5834c191afc32e293822d72
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2429bf0f5eaba75379ffa7eb0d210596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72ad31459f24d8927b948415081258b2
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ebc1dc27bd9fbd8345c84d6e9d53aeb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0abbffceca9d2af523b60f83df6a307c
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a2eef232883489590b2f677c93370f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfc02a2c33b8f367d2cdbbee1a98ee6b
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea06924d92425333dd0ad7583cc84557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c400c160e5834c191afc32e293822d72
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2429bf0f5eaba75379ffa7eb0d210596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72ad31459f24d8927b948415081258b2
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ebc1dc27bd9fbd8345c84d6e9d53aeb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0abbffceca9d2af523b60f83df6a307c
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a2eef232883489590b2f677c93370f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfc02a2c33b8f367d2cdbbee1a98ee6b
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_578b9cb8de1f6f0ff86954e5a0d541d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6858da4ccd9e87df4db75c152680e087
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d90b729e9c779d35f125a000aa70654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79d492005e5ad10afc7b231674cda7e9
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c87cd83fc90c41d48338ef37e3de1998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e99b9e2d091b4f081f81111ec3b8425d
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cc55219be8fc87f9587ee6d306b9ee0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e677e14b383b08c07760cb800dfa1caa
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62d130bc5cdb425ebda3eba70c32fbc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4bb927405d471c3245c40599e864ae0
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6d1d867cf43fa7a5aa16b36e7c9c0db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c400c160e5834c191afc32e293822d72
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99cefb534aa11ebc4a0afda45d8656a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72ad31459f24d8927b948415081258b2
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23579de53db0e3e6f461085bd5740c9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e677e14b383b08c07760cb800dfa1caa
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0207d017e21bd96447146d956a5c4327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4bb927405d471c3245c40599e864ae0
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ef02dc81dd80d8277f5e88a98910ad2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc01d7777bcc60aebfff17850f1d5e74
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5f161c596ae007d841cdd9ada7ed317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ad495e2146aa307d8d30704fbd2547a
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a05511a5caaaab4befe55dba5745274e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc01d7777bcc60aebfff17850f1d5e74
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5f161c596ae007d841cdd9ada7ed317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ad495e2146aa307d8d30704fbd2547a
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28ccca0c62ab5d0b22b49c403ed1ded5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6858da4ccd9e87df4db75c152680e087
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c68953c905382180fe1e3b0e30973631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79d492005e5ad10afc7b231674cda7e9
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f194be72af0bd5429c4ce0ebe22bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86ae2d67502ffd297c104c69b66347c6
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90507e31c569dad1f95ed3f5705da21c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_162ed8c85f2ae4b86307b7f30eb4c7a4
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34eec757428ca13a3b6a29a693f5c591(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 28, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d67bf6ea32a2ea508dc17743987e8ada(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34eec757428ca13a3b6a29a693f5c591
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e50792c554d32846f3ea0e34dcd4cce5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 240, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3519f05415d7a2c435e19c32239cfe25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e50792c554d32846f3ea0e34dcd4cce5
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c18abe94139f61527787f2c8d87717f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 10, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b01a6fb4dde0127296aaf0334666ebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c18abe94139f61527787f2c8d87717f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f818e2158de836dd630cbaae10c7cd12(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 672, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20488c359d28fbf07cd141cb965c8a3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f818e2158de836dd630cbaae10c7cd12
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d67bf6ea32a2ea508dc17743987e8ada(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34eec757428ca13a3b6a29a693f5c591
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7a46cd1ba87508cdb8b255ee36648cbd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 240, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80bd531aa0173b27dcba4e6010d9841f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a46cd1ba87508cdb8b255ee36648cbd
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b01a6fb4dde0127296aaf0334666ebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c18abe94139f61527787f2c8d87717f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45b67fe72447bffc3e95e4a986aa6aa7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 1152, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7be0bd9f8bec133602f02beef597c8f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45b67fe72447bffc3e95e4a986aa6aa7
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_24dcfd3c5412ac903aaa18ff38915ccb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce33196807b70108c60c468cc6395c79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24dcfd3c5412ac903aaa18ff38915ccb
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc390fde66f440583bbad6ed2df5ff48(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 672, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a594e5d471a81c166bd570d5a55e2014(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc390fde66f440583bbad6ed2df5ff48
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d67bf6ea32a2ea508dc17743987e8ada(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34eec757428ca13a3b6a29a693f5c591
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_493ca78b993baa1f237f746eb93b26b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ca0f6da8bfafa7dd9500e46ae1ebfe0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_493ca78b993baa1f237f746eb93b26b7
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e13e501031032bcc20a075fc856c7969(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 4, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a659f6178890be176aa65e62820af0e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e13e501031032bcc20a075fc856c7969
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_69d5aa684610d424f8bc9a360e6c9560(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 480, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29716ce93d0b93513d1081bbc1970121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69d5aa684610d424f8bc9a360e6c9560
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c1335d8b9dd9a4eee9cef45f3e4a48df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f71ae7d26cff8b0c25cafc8dbffb4599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1335d8b9dd9a4eee9cef45f3e4a48df
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0cf9ea6f41ccc9f47f683eb9852faf95(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 144, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61438a8a2d6b0901eea4f0cbbf876846(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cf9ea6f41ccc9f47f683eb9852faf95
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c1a6decb69360d377b5a4b37d681c2b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c7a46bcf134b501acc541e545c65c3b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a6decb69360d377b5a4b37d681c2b4
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80bd531aa0173b27dcba4e6010d9841f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a46cd1ba87508cdb8b255ee36648cbd
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b01a6fb4dde0127296aaf0334666ebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c18abe94139f61527787f2c8d87717f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_463a9e30190c6d806541e7bdc4b1af4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 144, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3836b184ca14eb3e125af73177879523(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463a9e30190c6d806541e7bdc4b1af4b
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2f1067edeca48bfdd0f46477623bc009(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a66f98fa83b110e7f13d1789973af6e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f1067edeca48bfdd0f46477623bc009
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0c9640b32387e0619bcc24b640a5967e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 672, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_def43ded7607f3f45b088355b918f095(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c9640b32387e0619bcc24b640a5967e
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1e3ef9216dabaa08b81e413ef04f32ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 28, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_619820de43f0c72db23c1fbd22fd9540(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e3ef9216dabaa08b81e413ef04f32ea
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29716ce93d0b93513d1081bbc1970121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69d5aa684610d424f8bc9a360e6c9560
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f71ae7d26cff8b0c25cafc8dbffb4599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1335d8b9dd9a4eee9cef45f3e4a48df
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3a53a592f6c8716b70a8744357ceabe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9fcf1ac760c13c6b962802eb65594510(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3a53a592f6c8716b70a8744357ceabe
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8123110dc3715853104e88c1f488dbca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 32, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3936da73a02f10df1e7838fea19b0b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8123110dc3715853104e88c1f488dbca
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9463ec85b70e9248e5f09b788eda8399(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 8, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6814d9a4cc61a8509a5762cfc143210f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9463ec85b70e9248e5f09b788eda8399
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d60d0a8ab76390d23bc5c39582ec908b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 480, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e276ddcc3570effe212417e090fdba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d60d0a8ab76390d23bc5c39582ec908b
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_23a9b4414a3cebf808090f1a9486ff4e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a354a1c8be81866dbebc8a742f33e7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23a9b4414a3cebf808090f1a9486ff4e
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_def43ded7607f3f45b088355b918f095(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0c9640b32387e0619bcc24b640a5967e
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619820de43f0c72db23c1fbd22fd9540(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e3ef9216dabaa08b81e413ef04f32ea
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_44f419decde7a66f5ddbda416cee07af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 144, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b897784f88d37001bb8b61dd894705b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44f419decde7a66f5ddbda416cee07af
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a66f98fa83b110e7f13d1789973af6e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f1067edeca48bfdd0f46477623bc009
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_160491b16af84e7b806325a5e395cdd9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 10, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d4c53d859b11bad6c7fceac29b0eb5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_160491b16af84e7b806325a5e395cdd9
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8f8e87a8e887a97c0dc2318d5012fa8e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 32, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b4910ff4248c9b2a3256caa5028b8e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f8e87a8e887a97c0dc2318d5012fa8e
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7c4ec04cb4d239c388d6d7516d03f26e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 8, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb4e1b212b8225fc5a48fc387b2854f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c4ec04cb4d239c388d6d7516d03f26e
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8d5cce3a5e313a708f61ddf6056e8ee6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 144, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51bd43c5fb23533da6b29f9105d7aaa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d5cce3a5e313a708f61ddf6056e8ee6
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7a46bcf134b501acc541e545c65c3b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a6decb69360d377b5a4b37d681c2b4
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7a46bcf134b501acc541e545c65c3b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a6decb69360d377b5a4b37d681c2b4
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cccee45d5b685b5677ec08a0386fbf44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 1152, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b40cd4de53497e5a0ba1eb7148de027d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cccee45d5b685b5677ec08a0386fbf44
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fcf1ac760c13c6b962802eb65594510(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3a53a592f6c8716b70a8744357ceabe
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b897784f88d37001bb8b61dd894705b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44f419decde7a66f5ddbda416cee07af
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a66f98fa83b110e7f13d1789973af6e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f1067edeca48bfdd0f46477623bc009
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d67bf6ea32a2ea508dc17743987e8ada(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34eec757428ca13a3b6a29a693f5c591
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3519f05415d7a2c435e19c32239cfe25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e50792c554d32846f3ea0e34dcd4cce5
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b01a6fb4dde0127296aaf0334666ebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c18abe94139f61527787f2c8d87717f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d4c53d859b11bad6c7fceac29b0eb5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_160491b16af84e7b806325a5e395cdd9
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_20f77ba67fc8278f3c288976c76c828e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c9bb3742c4f763c54a91fdcd7be8beed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20f77ba67fc8278f3c288976c76c828e
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fcd4d8cfdd888a2d87ba924107179543(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 4, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_722e2cdc24b7b36807fffe05e9312ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd4d8cfdd888a2d87ba924107179543
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c94b50876207ef88bca7ce517eccc37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 672, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3466b14f0c5fa9a51b90c6a6a10522fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c94b50876207ef88bca7ce517eccc37
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619820de43f0c72db23c1fbd22fd9540(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e3ef9216dabaa08b81e413ef04f32ea
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9bb3742c4f763c54a91fdcd7be8beed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20f77ba67fc8278f3c288976c76c828e
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_722e2cdc24b7b36807fffe05e9312ac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcd4d8cfdd888a2d87ba924107179543
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3466b14f0c5fa9a51b90c6a6a10522fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c94b50876207ef88bca7ce517eccc37
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_619820de43f0c72db23c1fbd22fd9540(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e3ef9216dabaa08b81e413ef04f32ea
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51bd43c5fb23533da6b29f9105d7aaa0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d5cce3a5e313a708f61ddf6056e8ee6
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7a46bcf134b501acc541e545c65c3b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a6decb69360d377b5a4b37d681c2b4
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a354a1c8be81866dbebc8a742f33e7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23a9b4414a3cebf808090f1a9486ff4e
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b4910ff4248c9b2a3256caa5028b8e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f8e87a8e887a97c0dc2318d5012fa8e
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb4e1b212b8225fc5a48fc387b2854f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c4ec04cb4d239c388d6d7516d03f26e
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ca0f6da8bfafa7dd9500e46ae1ebfe0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_493ca78b993baa1f237f746eb93b26b7
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a659f6178890be176aa65e62820af0e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e13e501031032bcc20a075fc856c7969
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3936da73a02f10df1e7838fea19b0b6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8123110dc3715853104e88c1f488dbca
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6814d9a4cc61a8509a5762cfc143210f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9463ec85b70e9248e5f09b788eda8399
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_35a6943d8845dab9513c3bacf2ec1b06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 240, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d8a6a71c1cd40b5c9813dd02164e70b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_35a6943d8845dab9513c3bacf2ec1b06
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d4c53d859b11bad6c7fceac29b0eb5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_160491b16af84e7b806325a5e395cdd9
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7145f30af156e59b2af9e8c53f2c0677(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 240, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71ea5ddf6e0eec80ee30463bf6229879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7145f30af156e59b2af9e8c53f2c0677
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d4c53d859b11bad6c7fceac29b0eb5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_160491b16af84e7b806325a5e395cdd9
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3836b184ca14eb3e125af73177879523(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_463a9e30190c6d806541e7bdc4b1af4b
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a66f98fa83b110e7f13d1789973af6e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f1067edeca48bfdd0f46477623bc009
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7be0bd9f8bec133602f02beef597c8f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45b67fe72447bffc3e95e4a986aa6aa7
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce33196807b70108c60c468cc6395c79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24dcfd3c5412ac903aaa18ff38915ccb
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d47d93ccd3b7d02bfc4ca782b95e0d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ef645fbe9e14669981886a4e5a099a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae5986207c66d9aee232e37682cb97b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbc4bdfa81c1c3696d7430b8e93389ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d47d93ccd3b7d02bfc4ca782b95e0d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3895fd1d74204e8865b43871a50cd7b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae5986207c66d9aee232e37682cb97b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad5e7f093c3eebcb196b85458522cdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e10a20597658a5c630602a27f4609e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a402c901db0acd7984434c3bc7c5f49e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d47d93ccd3b7d02bfc4ca782b95e0d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce3d5ecca025de35aa99e40beed0a4c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42be5be354e33d5402ea07d86540f6d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cdf744bee8a2ac799de2658b0a1868d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a87735d377ea1209e720a639a6d4e8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cb0a4c345a7ebda5278286abc9f7fd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb6d110cb83d37e4466808f601dcea2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3895fd1d74204e8865b43871a50cd7b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae5986207c66d9aee232e37682cb97b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac3c0ddcd2fad788837781edd39a958b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e56f8c8412743282968c88ae5a03216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0beeae12af7ac74dd7a72d5884715f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10edad9daa6bfa9e29a156b4cfb67545(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cdf744bee8a2ac799de2658b0a1868d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a87735d377ea1209e720a639a6d4e8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8b8972007b4945625b7f29fcedff533(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638b1874f1a96714df8844b4ea509630(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec091ecc66080975c515417955f3b407(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_563ddb82aa78374124051535d42fda31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a3eddcd2075845e6f570ff0246d4097(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0beeae12af7ac74dd7a72d5884715f8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10edad9daa6bfa9e29a156b4cfb67545(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b91cf11c39e9013c9ad19bac56e10ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e56f8c8412743282968c88ae5a03216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_725177757e3eb153f0177ddc1ed4b5fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_387dba6fbdd5325a479f4e090588b5f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e51fb6fca556f2834d46c4934f39853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6fd039616478beb40ce510457701702(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb6d110cb83d37e4466808f601dcea2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb6d110cb83d37e4466808f601dcea2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e3f12824f7c1614e78e7c5ed2431f5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8b8972007b4945625b7f29fcedff533(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b91cf11c39e9013c9ad19bac56e10ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e56f8c8412743282968c88ae5a03216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d47d93ccd3b7d02bfc4ca782b95e0d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ef645fbe9e14669981886a4e5a099a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae5986207c66d9aee232e37682cb97b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_725177757e3eb153f0177ddc1ed4b5fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7df3404d179fcd441e07b3ffdccaaa12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23d64eb6934167e9073c06bdbd4fdd23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b605fbfbdf43885270c64f933a9265e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10edad9daa6bfa9e29a156b4cfb67545(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7df3404d179fcd441e07b3ffdccaaa12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23d64eb6934167e9073c06bdbd4fdd23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b605fbfbdf43885270c64f933a9265e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10edad9daa6bfa9e29a156b4cfb67545(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6fd039616478beb40ce510457701702(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb6d110cb83d37e4466808f601dcea2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a3eddcd2075845e6f570ff0246d4097(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_387dba6fbdd5325a479f4e090588b5f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e51fb6fca556f2834d46c4934f39853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce3d5ecca025de35aa99e40beed0a4c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42be5be354e33d5402ea07d86540f6d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_638b1874f1a96714df8844b4ea509630(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec091ecc66080975c515417955f3b407(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1160cfb4c8f37e7af707d332cd228e52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_725177757e3eb153f0177ddc1ed4b5fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_573d2e2fb67f481ac9f22a8f7d6d0ace(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_725177757e3eb153f0177ddc1ed4b5fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac3c0ddcd2fad788837781edd39a958b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e56f8c8412743282968c88ae5a03216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad5e7f093c3eebcb196b85458522cdc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e10a20597658a5c630602a27f4609e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13bf71d674ecd60661e5204fb9b48a0e
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()