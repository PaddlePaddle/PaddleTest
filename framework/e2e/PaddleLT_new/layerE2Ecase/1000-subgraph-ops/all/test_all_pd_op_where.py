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
    class PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e083bcdf8923ac7571d411e6f3deb979(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1024, 5], dtype='int32'), 'bool'),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6087ec89a7d071e1921f556ada3be750(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2100], dtype='bool'),
                paddle.static.InputSpec(shape=[None, 2100], dtype='int32'),
                paddle.static.InputSpec(shape=[None, 2100], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e9cec67a835db3037edc07e93e91f3f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6087ec89a7d071e1921f556ada3be750
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_91b951befb03af2b9789137a1742b56d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[4096, 5], dtype='int32'), 'bool'),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='bool'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2605e189271a849bb54b4b33b1354ff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_2605e189271a849bb54b4b33b1354ff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_486c7e862f457e586d74f4b9feee7a83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1021], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_486c7e862f457e586d74f4b9feee7a83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1021], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_56ad9a5ec68acf04818498fbb763b08e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_56ad9a5ec68acf04818498fbb763b08e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_2b4750e1adbb36ee08d3830e54da1f15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[64, 5], dtype='int32'), 'bool'),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_730b47f8c3aa535440c0268376fdb2a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3549], dtype='bool'),
                paddle.static.InputSpec(shape=[None, 3549], dtype='int32'),
                paddle.static.InputSpec(shape=[None, 3549], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d9a5c2d1ebbac5d08f7c6c68fafab23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_730b47f8c3aa535440c0268376fdb2a1
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    
    class PrimitiveOp_83ce4da179d0e03615887046b0ec02a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4116], dtype='bool'),
                paddle.static.InputSpec(shape=[None, 4116], dtype='int32'),
                paddle.static.InputSpec(shape=[None, 4116], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e9ffd13fe564e72cd7421ddcffee6df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83ce4da179d0e03615887046b0ec02a1
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_f4195fbe48f2fcdf263f4eb23f6fabc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[16384, 5], dtype='int32'), 'bool'),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de96bbb5796261dc78166f023acb17e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1027], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_de96bbb5796261dc78166f023acb17e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1027], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_f45523b0cc914665fbac41870475ae47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[256, 5], dtype='int32'), 'bool'),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d61c781a30426822b587593ca1d1d316(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1024, 5], dtype='bool'),
                paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_80ea2ee2e3bbe3fb5ea769fbac6e37d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d61c781a30426822b587593ca1d1d316
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1024, 5], dtype='int32'), 'bool'),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f8d3441599f78dba3017c948f6f57254(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='bool'),
                paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
                paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f52ef985043dd6902e813ad421caf6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8d3441599f78dba3017c948f6f57254
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    
    class PrimitiveOp_3d3b8ecf309477689364d4e74b11bc50(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4096, 5], dtype='bool'),
                paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0909b8a258545feac49747bd46678f92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3d3b8ecf309477689364d4e74b11bc50
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[4096, 5], dtype='int32'), 'bool'),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0be6c0b0b0e52055a728e513e6a3407e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2002], dtype='bool'),
                paddle.static.InputSpec(shape=[2002], dtype='int32'),
                paddle.static.InputSpec(shape=[2002], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f679b3200b32da99f62291135748e2a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0be6c0b0b0e52055a728e513e6a3407e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_f679b3200b32da99f62291135748e2a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0be6c0b0b0e52055a728e513e6a3407e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    
    class PrimitiveOp_944df9faabcf139a308ad5a7603d7cad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1021], dtype='bool'),
                paddle.static.InputSpec(shape=[1021], dtype='int32'),
                paddle.static.InputSpec(shape=[1021], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75ca002f306c83117891f3e880fb03c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944df9faabcf139a308ad5a7603d7cad
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1021], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_75ca002f306c83117891f3e880fb03c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_944df9faabcf139a308ad5a7603d7cad
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1021], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    
    class PrimitiveOp_26b42d4e5f70cd17a2772550963fa62c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1002], dtype='bool'),
                paddle.static.InputSpec(shape=[1002], dtype='int32'),
                paddle.static.InputSpec(shape=[1002], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b1bc5bb785c1a055999e03b4a52517c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26b42d4e5f70cd17a2772550963fa62c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_8b1bc5bb785c1a055999e03b4a52517c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26b42d4e5f70cd17a2772550963fa62c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    
    class PrimitiveOp_f228ca65d9915a77172b0261a78740f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 5], dtype='bool'),
                paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71142afa2a9aa94938f6cad71b650393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f228ca65d9915a77172b0261a78740f4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[64, 5], dtype='int32'), 'bool'),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0df2a5f913dbdede71e6023cca7bb6f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='bool'),
                paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
                paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5a467f3340bd53e25460c167bb4e82f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0df2a5f913dbdede71e6023cca7bb6f5
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    
    class PrimitiveOp_31aebadd717e036d7fd0939b0897447a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='bool'),
                paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
                paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf2e97e9d5407e8ac1340b94e11e4f07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31aebadd717e036d7fd0939b0897447a
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    
    class PrimitiveOp_70b2fcdffef6362f23dfb8621899132e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16384, 5], dtype='bool'),
                paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29cb4879feab68df69cdcaed89a281f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70b2fcdffef6362f23dfb8621899132e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[16384, 5], dtype='int32'), 'bool'),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_96eb96db5074a6eea33dd60c241f31f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1027], dtype='bool'),
                paddle.static.InputSpec(shape=[1027], dtype='int32'),
                paddle.static.InputSpec(shape=[1027], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a705f323a1c114a97e11fb5b1bf2597c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96eb96db5074a6eea33dd60c241f31f0
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1027], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_a705f323a1c114a97e11fb5b1bf2597c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_96eb96db5074a6eea33dd60c241f31f0
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1027], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    
    class PrimitiveOp_acbb738db013e511320046bc95a8aed8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[256, 5], dtype='bool'),
                paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
                paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9622b9afde28a51305e51f2ce449266f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acbb738db013e511320046bc95a8aed8
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[256, 5], dtype='int32'), 'bool'),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e083bcdf8923ac7571d411e6f3deb979(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1024, 5], dtype='int32'), 'bool'),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7b64aaa92c190929935d29b268dafa86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.where(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_012da10620d0b6ad543f348f02d8d346(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b64aaa92c190929935d29b268dafa86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_91b951befb03af2b9789137a1742b56d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[4096, 5], dtype='int32'), 'bool'),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2605e189271a849bb54b4b33b1354ff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_2605e189271a849bb54b4b33b1354ff3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
            ]


    class TestPrimitiveOp_486c7e862f457e586d74f4b9feee7a83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1021], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_486c7e862f457e586d74f4b9feee7a83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1021], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
            ]


    class TestPrimitiveOp_56ad9a5ec68acf04818498fbb763b08e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_56ad9a5ec68acf04818498fbb763b08e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1002], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
            ]


    class TestPrimitiveOp_2b4750e1adbb36ee08d3830e54da1f15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[64, 5], dtype='int32'), 'bool'),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1923503b9228c74b8e00d6942c6110ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b64aaa92c190929935d29b268dafa86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_b2cb22779b5c495f04bca785d824b2fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b64aaa92c190929935d29b268dafa86
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_f4195fbe48f2fcdf263f4eb23f6fabc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[16384, 5], dtype='int32'), 'bool'),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de96bbb5796261dc78166f023acb17e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1027], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_de96bbb5796261dc78166f023acb17e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56a5b329c79ffa332d0b9901394b9e56
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1027], dtype='int32'), 'bool'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
            ]


    class TestPrimitiveOp_f45523b0cc914665fbac41870475ae47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfb7e1021f9e33e86eb372dd26553821
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[256, 5], dtype='int32'), 'bool'),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()