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
    class PrimitiveOp_d79a1eeca5d0a8f22939d67abc66b382(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11a92c4a308bb41624499152850699da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d79a1eeca5d0a8f22939d67abc66b382
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0f32d512811e3d805f76efed5e2d765(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ae2c939cf103bf1dd05273b097b3036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96fa52cddfdfb197dcd6cea697c05fcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_025d654729e7a9c033a53c4dd9186ea1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8586b0f4d671b282c422bfc77353afd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_025d654729e7a9c033a53c4dd9186ea1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cc5451a3cb06175857659337311712b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebca71a2b6be1ba735a0ac0e38938d06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c33be5521df8badff101415bcb88a6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4da1d2d51b401f95e960a6df566faf06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3665f2806f8b13c76c77173649f28d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_848f62ce2e6b5439e31fb274ee91fd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fa2af8ab3ebc9208f0278aa07ace4cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5582ee8dcef92d93aa7a9115386b9f5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [5, 5]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [2, 2], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28bc57ca7e07569e428c5c18bb2dc51a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f30f70b5785796993b6a405b63a27f3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea91e980738718e157b91369008a2cf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0d536db09f4dec98d6533b073390b31c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [13, 13]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [6, 6], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af8d935f9681f6410c76d55b4c354387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_482f173584775af2077d4452df8f425d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1636c00c0fb2d3c59df6cc205ec54854(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8180beddade27c19a2e7edc810e7c9ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1636c00c0fb2d3c59df6cc205ec54854
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4b9a8701dd43d50f9740b783464dbae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a715f654bdd5232a8b417d9700a8d26b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4b9a8701dd43d50f9740b783464dbae
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_50e7941ad1f4c3d965da50260dc17822(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cedcce8cb7cbeaf57382e26b2e35bec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50e7941ad1f4c3d965da50260dc17822
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6ebb414d7953159cbb789766562992de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_35d7aed3ec981f0983097c708eeb1e2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2b918368d2079a8b487ad9485d44a8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c6348a86ea2d64041b7e751598a345b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55de72e92288231e22f84795eae485fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_28cb794c381baf9658d1141320a4b9d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f4ff6101c37af55337b4f1785db166c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28cb794c381baf9658d1141320a4b9d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c838f098e020cd698c8dcd91c80e042(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50e7941ad1f4c3d965da50260dc17822
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c87ef396c8878c545d20a85d3148e79f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e6f24900b11f55f34ed87c301f444120(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1152, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6255daa051e91b7951117763166b81af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f24900b11f55f34ed87c301f444120
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af7287320b3e04f046f0d398573fef26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8da12591224934238b754b7af318f19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa4feb011790a997b3902d564c8ea4de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57836e1df3a60efffb8557662ad31f15(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b880a3107bed0efaf912672a53c2c5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_401ca27f95b72e59daae30320099cf2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a85571244114b9ab6cf6926d01556346(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abea680ad996fa26222a5ba1846bcd11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ae9cefa290d24f2052b760ba22dfa92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84d1be8a07de93c67081950bf7ba2072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ae9cefa290d24f2052b760ba22dfa92
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aafcf76ced1a0ece5055e6a3df448831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8402f6e2460324f32f79179a49d94724(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 44, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88a9c09c50e2be636ef5050c15650920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8402f6e2460324f32f79179a49d94724
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2f2728e26890922c78203a39640d0ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_025d654729e7a9c033a53c4dd9186ea1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f7c6cbab1995b46978d1bc21f71ed42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50e7941ad1f4c3d965da50260dc17822
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c957d4fa5ecb9cea92b160f225fcafb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1024, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39979e68093212a5fbf5c323a620f99d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c957d4fa5ecb9cea92b160f225fcafb
        def get_inputs(self):
            return [
                paddle.uniform([10, 1024, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_246f5290018ba323eea342f30f2d5b72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb75dbc4067ee1009a94e0056aedee0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62c4a5fc79a5ea515224f68b83724cfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0590a6b77b26f531f5d95c8d007d24a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c45934fc4c09883beede8d6be7e04aa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21bfc94dd5de1d3843a1c1b5bfff61d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c79d64088dab7086c934cd406eb10d88(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04c5ae28f9597e6fb82d9fdeb5814ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c79d64088dab7086c934cd406eb10d88
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1973767dae8b91ff3512b3a8b23a5ced(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f649cde71839b12f4f10761c36c63e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1973767dae8b91ff3512b3a8b23a5ced
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0bc3b8a422b8373062a1a669f09c25d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62d68cf21f808e51b8e9f87a8ec0458f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bc3b8a422b8373062a1a669f09c25d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a74510b27a6d14e8e0fd64d72003e2be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4744b6a2104884b1f9debdbde9c50d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74510b27a6d14e8e0fd64d72003e2be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3912524e9d06ecced673341957aad496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0256cd40851a0c0f8cf8a99854b42005(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4d46110686466903cf24c07f354fa71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81dc63d9ae1942666fb3004159f178c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_795099eb19ce3e66ceb068586075608a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04617d9ed4eee2a91a7568ea1cd8d5c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795099eb19ce3e66ceb068586075608a
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5146c72c64a75c9bf81bf26461a87c25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [16, 16]
            return paddle._C_ops.pool2d(input_0, input_1, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f4ab66901df3f31bbeeed5b1045e5dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5146c72c64a75c9bf81bf26461a87c25
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e326024d4638b71fe1319d583a48088d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [8, 8]
            return paddle._C_ops.pool2d(input_0, input_1, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82d81b08a48e737c919d54748631a0e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e326024d4638b71fe1319d583a48088d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4dcb175a35f23da7400f46656efba6a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [4, 4]
            return paddle._C_ops.pool2d(input_0, input_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41d5cf3fc6cdd53596663b79a887bd3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcb175a35f23da7400f46656efba6a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2659142df9490b914751e078ad7a6c0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_087914b58e74f540d7155e4b2fcc1920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03c540c13f4c8a62e6fbd12c806b50f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5146c72c64a75c9bf81bf26461a87c25
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35b5e968fe4b26375ee1c9113750ca24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e326024d4638b71fe1319d583a48088d
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93caf0eb0998531ca496255a1c0d145a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcb175a35f23da7400f46656efba6a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cd553d7ceabd7f6b4b7e5c57dad9f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_715127df718e4222bf5858fe9c5c0fd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_800dc3d644cb4513585fa6b9e9ee7948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4da1d2d51b401f95e960a6df566faf06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_14c6d0564c1149c8bb15622289c108b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_66b2509d094903b3083f47a0763e99de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14c6d0564c1149c8bb15622289c108b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c2998e2a920afa6250fa8ddc1d17a47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e85cfecdc89d4fb415ce7b5f2ec48038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eb15bb06bb723bd0fe291dd8f819bada(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8fb81abf45f87336296ea8057a36bc09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb15bb06bb723bd0fe291dd8f819bada
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07969d7e81a54dba7e46f0fa3c629b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6d131046c72e941483437a0faa1a4c35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fbab95cbfe0b22788a56c795513d6149(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d131046c72e941483437a0faa1a4c35
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca8cd086d0b6e463cf2186166f86d6b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50e7941ad1f4c3d965da50260dc17822
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54c97c46e462d172b2fb99741b216835(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33346bb59fb104cf3b0bbd59c13b8301(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a9512cfd9326ad56f25628b10b6a16a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8822ae65c00cebcf5ad673f6b7a8ebe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8259562aebe9695d941c29763a5ab779(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_247d8331dda7c5b1a0b0a2e923ad250a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8259562aebe9695d941c29763a5ab779
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_647f64847ee964a8b4d495afc30dff1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b2781c1916ee52aa95a418a9cb844fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c4f4d04de0043df773bd3dc3c8293993(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9234e92a4a5e8286a8f387e4af7a36bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4f4d04de0043df773bd3dc3c8293993
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4bc4ed0652d278f82c8d1a7a069c8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_258821600d69f3ec2aa05f4c412503b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, 1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3011ca7004c107ae52191602cef6172d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258821600d69f3ec2aa05f4c412503b5
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc6d99f0d675d255326d55ddb4145b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb15bb06bb723bd0fe291dd8f819bada
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37dce169684ef65ce1a7a4eabbcd4dcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8db402d6a7d5af32522453e5475d81a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff01e493739eba9a29970ae5c5766ba3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28cb794c381baf9658d1141320a4b9d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_493f101d25a7457f66f801bb2ca0ad39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c73387b024d26d4480319eb26fa82f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db21b58b36265d59be1e4396210501f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea45396a9ac535075999b16ef0090dd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d79a1eeca5d0a8f22939d67abc66b382
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e4cacb62c4787135b99e1d056d492d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db1eb740f6bae2991b6a3e1a87fe4c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae1e9a7fed6796a735fa0431825f4e8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6165c68500f3ff8222e5207922a54f6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df1b21a9d7d0c132774d25cbae91df01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 320, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d8308a2f7c45c5565753eface6e4c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2ce7040eca8cbab9d1d92b0db42fb275(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a09070c64848f8bad7645d017ffb3bfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ce7040eca8cbab9d1d92b0db42fb275
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8480007012f4325d31c1a5a3062524d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc28c7828092540d62218b56247dd435(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8480007012f4325d31c1a5a3062524d
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0050a12f0a3e88c9b56608fa9feec33d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f791ba1ccdc4c7c3cb7c8b75f24e425(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0050a12f0a3e88c9b56608fa9feec33d
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7a02a78343c3ba6a629c90998775d683(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0cee00caad000a3516a9bd71ae8f6c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a02a78343c3ba6a629c90998775d683
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d779beb61fc60d8cc2392b620b5a0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df76d17075a467b5fc4b740b45e7b9bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28cb794c381baf9658d1141320a4b9d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d50cbdd1363624833f35227891391278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9d4f9f64a5bb1ced13fb77c340be635(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8f6b5a92a16c61cd2d323860271d38f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28cb794c381baf9658d1141320a4b9d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a97246081f67fe772e8376b379caebdb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71b0eabca2863099fec90fd570db87cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97246081f67fe772e8376b379caebdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a5258fee208d2b17250f1bb0c937572(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08bf7dc410899192afb2ce83ff604433(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_437d99a34deec895b51fb40073829db9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28cb794c381baf9658d1141320a4b9d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_098f7f7df21c1e9c7842352f38db940d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97246081f67fe772e8376b379caebdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39a3125508c27c5e3131979ad7570379(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c87ef396c8878c545d20a85d3148e79f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cb422e31b0eba88f1452341106222eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27f3fd07554692dcd1e087fd6b398ec4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ce7040eca8cbab9d1d92b0db42fb275
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a2450cbfe3397f13fd4becc2e8c2b7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8480007012f4325d31c1a5a3062524d
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5169d09fb1f86cfeb92c9c2ac04cf597(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0050a12f0a3e88c9b56608fa9feec33d
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db6b466b34618e90367fef7b2b096349(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a02a78343c3ba6a629c90998775d683
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f3baa90f6de1192845ec833c45932b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6da0e8529c2fc1d8bab0b1bdfda62d58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38f7345605a7df26a84c5dbc152250f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_622f62ff4d2ba8e42253df084b1794eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_31a406ffed699a46e8f41bfc8bd31cc0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_705ae6bad845802d49f5e89bbb9b6e77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a406ffed699a46e8f41bfc8bd31cc0
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_705ae6bad845802d49f5e89bbb9b6e77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a406ffed699a46e8f41bfc8bd31cc0
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a5b94915dee2f9411cb70b18f789550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a406ffed699a46e8f41bfc8bd31cc0
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a5b94915dee2f9411cb70b18f789550(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a406ffed699a46e8f41bfc8bd31cc0
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f36e9516d5f0b856c6a35c0a908d3763(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a406ffed699a46e8f41bfc8bd31cc0
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f36e9516d5f0b856c6a35c0a908d3763(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a406ffed699a46e8f41bfc8bd31cc0
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_028776c7ead24c1f384b52ad6630fcd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a406ffed699a46e8f41bfc8bd31cc0
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_028776c7ead24c1f384b52ad6630fcd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31a406ffed699a46e8f41bfc8bd31cc0
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f05430c372a86dd09753884e650b50a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ee04f807058718f86d217ecc15656456(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a597e6289e3714b2e6db824607d22f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a597e6289e3714b2e6db824607d22f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3760b70cf3d0db0e18a9016c4b088f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3760b70cf3d0db0e18a9016c4b088f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fded6ae55932f726c82d6a3aef6fd7b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fded6ae55932f726c82d6a3aef6fd7b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4a8a010adfd4d39551fc633d66f078f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4a8a010adfd4d39551fc633d66f078f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90469da5d2040aaded843b467d1def9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b3abbbab3a02b361d66d6c041aaae8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14c6d0564c1149c8bb15622289c108b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e4cacb62c4787135b99e1d056d492d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab367c591716a202d79e191933b90b4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6cd0dc8a2bebb2e7c4f43511c6fd1f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f24900b11f55f34ed87c301f444120
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da94eb11545b92d4703e28231f737ee0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccd252dd17cb6416f44b55305b80111c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da94eb11545b92d4703e28231f737ee0
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ea39a4896348a1de14b1632616f0999(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27070c35c8e359c551ec0f217bd04156(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2431fb051c6365386ee9fe9adb75a3a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_460420239d6181c29fd3d56eeb6dd825(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [14, 14]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ada5f439764ab6cf1f9370fe7415b9c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_460420239d6181c29fd3d56eeb6dd825
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ada5f439764ab6cf1f9370fe7415b9c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_460420239d6181c29fd3d56eeb6dd825
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4eebd1ed894757319eb3c57183451782(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_460420239d6181c29fd3d56eeb6dd825
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4eebd1ed894757319eb3c57183451782(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_460420239d6181c29fd3d56eeb6dd825
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275a5c4257eee5e7c2207607649b377b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_460420239d6181c29fd3d56eeb6dd825
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275a5c4257eee5e7c2207607649b377b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_460420239d6181c29fd3d56eeb6dd825
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acb117a5910cf5ddb80027361df224ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_460420239d6181c29fd3d56eeb6dd825
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acb117a5910cf5ddb80027361df224ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_460420239d6181c29fd3d56eeb6dd825
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d93d0c90108b90a8091d687dd8f84075(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d319ada71de6bce25478e85a75f8d8ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d93d0c90108b90a8091d687dd8f84075
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_edcce8e1da01c895cfc2084de2a15b6f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44d2dac5891beae4d938e0590d099127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_edcce8e1da01c895cfc2084de2a15b6f
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cb21d4708a0216e78e2aa4554fcd39ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5553cae9723dd242f581eaa306fb6fe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb21d4708a0216e78e2aa4554fcd39ee
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cffec4fb2ae93752b9820beae30f212c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_781833349d0a55860f4bf8309b486cf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cffec4fb2ae93752b9820beae30f212c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f8b970dd01d82ae95324e38aa8e8285(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c2fa5d7a91f7aea139ef7b1dc5cf356(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f8b970dd01d82ae95324e38aa8e8285
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a27ce69f4eec6b4cd6b66046bb15c765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ccf14af7082c121de41640e99c9c0f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfd9f6b2668ff049202394b8f1b3938b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8e2cb8168c12341cb9620c67a7b6c4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c79d64088dab7086c934cd406eb10d88
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c30bf0153f7cd0b7491e036ec1135f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1973767dae8b91ff3512b3a8b23a5ced
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73b8e0d075189edd5eb781d5f67c54e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bc3b8a422b8373062a1a669f09c25d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54a7b1653746b70d9b8b2c40af189e0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74510b27a6d14e8e0fd64d72003e2be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38a0ac9253366bc0469f68ddb2e56219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da4ec050f29cc73fdb99ad0a74a6d16a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_609f981ae27f65b618bff28afb844641(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28cb794c381baf9658d1141320a4b9d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75bcb90e13d694dddf914b93818d908c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cfd33f1f27c4d7aaaa96e96e85f5cc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49726681b862f2ed4f599372a7ed18d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NHWC', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7291d7fbd3737132da715adba7afde4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49726681b862f2ed4f599372a7ed18d5
        def get_inputs(self):
            return [
                paddle.uniform([22, 7, 7, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_817ce72b6a1958c84ff5b0f226fbf3eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa3e8fc6a5672ba7e9fc2f3e501b251a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f25b5708f0bd139cb07fec5924a5284f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_118a76d9b1e4f2f9757b74c94bc74bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_025d654729e7a9c033a53c4dd9186ea1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a6317d1a01283b8d84a7bb962547d64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795099eb19ce3e66ceb068586075608a
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3c7aee812d2a38a4db14967b6f885f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be076c3d3020660e1e995aa1b213e121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44accfd304df6315a27bad63ff2436f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f05430c372a86dd09753884e650b50a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd1ae884c99d38ce83d77ee5738a86d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4abc0ab1b9f0a1570c8accb8ccca9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96b04aeea79df938b3e44d96f4df77d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28cb794c381baf9658d1141320a4b9d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6dedb105b83f8768f7ac4c22fc7074a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28b394544f96c72754b787a43e960e64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ae9cefa290d24f2052b760ba22dfa92
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7400bf86bfda1fb2c1e764bcb37b42bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cfcac9b7b5c6745bf91f5e76fb6e6bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f47192ee2a30c09efd1da2c358b62bc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a618c09ec174f1bfad35df251d9b079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ce7040eca8cbab9d1d92b0db42fb275
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_312c205a85c3e631cb5bcde29bc56504(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8480007012f4325d31c1a5a3062524d
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2457ca4d399c85bebf3b0812f3977302(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0050a12f0a3e88c9b56608fa9feec33d
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92683c614f4e16c3cb789230ba3b2eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a02a78343c3ba6a629c90998775d683
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_504ee919b5a2a9bc1dd7e5179f60ea36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78a254f17f7d6e0640d480b8b009a00e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8259562aebe9695d941c29763a5ab779
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a5f6005d15a5344cd217aec8c10ae6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac84c305c3fca971c622c8a8446a940c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59de7179a611a4c3dbef82f322f52e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_36180af9cf18b32f66f49f2a46fa317d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 400, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbe14831c0c971d3ba326c1eab416192(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36180af9cf18b32f66f49f2a46fa317d
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60cc23cf3e7d01f8c3d98dc8ff0f6ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ac243ef43a943343fb706d79e43d107d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3fa0884144067255b969bf162423db14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac243ef43a943343fb706d79e43d107d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86265dbcc230fe543fec14011f3f032a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2db4b97d4b10db81b910927c287bcdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc55e5a03b08523fbc110d37ecf919b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e41c4d355cc5027d0ecd04d0e1b53ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_937da3af0d78d866884b4cf1e9e76853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bcbe6ff2989a4a01f4c9a62aa0d869e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af94c6f977d64194b5715d2bed3317c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 38, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3bf773429da85e0e35a2bbd5e0911d7b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [9, 9]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [4, 4], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_adfae89d34c09456e83986e018613310(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bf773429da85e0e35a2bbd5e0911d7b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2364f1e0b8c817ff498a6fb14cd09f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79e80dd6be753ec7c1132c79e404d4d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97246081f67fe772e8376b379caebdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2931718cd538cc4d4da9a14c8eb33c0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5f830a2bcb20f93d6eff7910d287838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4f4d04de0043df773bd3dc3c8293993
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daf631cc05e557b406e60e91e34a631b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f021d38aebcccbc1f7c97c4c59368cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d6b75bf600dd1d0a5591b56086bce98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 320, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1880dd9028c02f048b92211fb5dad5f6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 336, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a877d7c2dc6f64c2adfa42dc323df715(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1880dd9028c02f048b92211fb5dad5f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac1517476ab269d8ae75057877516369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da94eb11545b92d4703e28231f737ee0
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a079ca501124e63f85d1444bba22fe3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_11acc5f78ffdacb9b9711c3a4545d8be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbfe5bbe95095c422ca25bdd5cce3f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acc5f78ffdacb9b9711c3a4545d8be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ff6ee206d2058659644c5653c81bc99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4f4d04de0043df773bd3dc3c8293993
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_499ac92eaa80d292775e273aac69b3ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8402f6e2460324f32f79179a49d94724
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfe52412ac19b068c189eb791125e131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8f61060ecc57e00fdeb26e4a1ee9d20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c957d4fa5ecb9cea92b160f225fcafb
        def get_inputs(self):
            return [
                paddle.uniform([22, 1024, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d535f471cb1eb23a60def8b6f0bcd7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36180af9cf18b32f66f49f2a46fa317d
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_28fea17e1de9fab9b6a824617b1a84d2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 56, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4624232e53af90d69a712949840c2622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea17e1de9fab9b6a824617b1a84d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60cc23cf3e7d01f8c3d98dc8ff0f6ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0a41c66d4fb05740fc8648cdf922d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bf773429da85e0e35a2bbd5e0911d7b
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_27ebb52474a7c58865cf912c53aabf9d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bef16cc7d6ba6047f8223ad59f448efb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27ebb52474a7c58865cf912c53aabf9d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7ded7d63d3b12e73853c86d616b2d31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_025d654729e7a9c033a53c4dd9186ea1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85678aff6dd41a916e91bb2e1bae83c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4f4d04de0043df773bd3dc3c8293993
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a5258fee208d2b17250f1bb0c937572(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fdb2c51bdfe62ff77748bec0e54eb96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b855e8ac8930ab8ff1fb8c9b6f242f84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61f774269e68598e6f114ef57842348e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9178e92db2041db3faf738ae79286404(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a6e626736f7189803d672bf24eb89903(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2ddb109fc8cd80dbbe533752b171079e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e626736f7189803d672bf24eb89903
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39fd4ff1e7acbf28077468a1b010de12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e626736f7189803d672bf24eb89903
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b1185df4f4c494980043b4918667001(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e626736f7189803d672bf24eb89903
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d74b1e9dc027e85a53a7b800e90518c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e626736f7189803d672bf24eb89903
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6cd0dc8a2bebb2e7c4f43511c6fd1f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f24900b11f55f34ed87c301f444120
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bad619877f259d7b3cee0110e4f70f62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6dedb105b83f8768f7ac4c22fc7074a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e121ccb5b04b6a33c13a2823a0eb70d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a97246081f67fe772e8376b379caebdb
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4678365ea3082d602e4db653712d67f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd88c866e6c4dab1aa97fbbb9c4cfdfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b454cfc7da82c77b3d8d69a2025fb535(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_895349c381382a05f9c4e2e5691e581b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_258821600d69f3ec2aa05f4c412503b5
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b925cc2cd4a91157b9ce9350fa450fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 200, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbff1d13e09c0e55c2ba9285e02e69c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b925cc2cd4a91157b9ce9350fa450fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aadc667b417bb63a7d19ab58c453f3f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba287ee42b934b7797120582248386c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89e91ce9c4d3c3f21a2c38c29ad70310(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17f634a4d45f1eb13cfb5db5a6317b08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acc5f78ffdacb9b9711c3a4545d8be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cf8c3d5e6b6ee3cdb09dc6ec6ada838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acc5f78ffdacb9b9711c3a4545d8be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55de72e92288231e22f84795eae485fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_29cc8888b8a6597ea31bed8fab1f07d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [7, 7], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7bbe30f7d9816168cb0eadb1284cc317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29cc8888b8a6597ea31bed8fab1f07d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2b918368d2079a8b487ad9485d44a8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39601286b445de3f58574facc5798aa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d131046c72e941483437a0faa1a4c35
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_65e7882c27d74ef5d27a343485f0edd5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d1b0112a95bbe0c9a98cea4c39cd5b9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65e7882c27d74ef5d27a343485f0edd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e02278a8c598994aed83f90ef6ae6a5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc22fc6f02d137c2fee321dc3a35df68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c65f350b6c01a40597bfde63db77d435(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c869d91a703c2554ff3344f33d13e79c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c869d91a703c2554ff3344f33d13e79c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f44800241b02123f95f187d7fbd53931(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f44800241b02123f95f187d7fbd53931(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3515e8917129bbdfad4dc658dbff8ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3515e8917129bbdfad4dc658dbff8ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e31b5d171ddf8af80c66fc6f227dc9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e31b5d171ddf8af80c66fc6f227dc9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee04f807058718f86d217ecc15656456
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ead436e7c1721ec5e91f4658b947823e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28cb794c381baf9658d1141320a4b9d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82ea6b6eee7e1e601a3e33370657098a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bcd3258996c4f88b7b55880a54f841e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f511badd00a66784a0ca07f2a1e34ea4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_341f8986821ed9eccf10d581ad82151c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fca81d5c59a2b19298e4d79b9d9698c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2629c39003f3c8703643fc53d26347d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cb69282d09833b5ffdeba56919e8976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a36bf7166fc826e8a8165aeb3135e2ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_929e9eb7e5ee31f2f38370e29cd401d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5146c72c64a75c9bf81bf26461a87c25
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e966c5edb1ec242b0858537b4f9ae6f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e326024d4638b71fe1319d583a48088d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffdc5ad1af9fe0f0c6992f01d16fe8e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcb175a35f23da7400f46656efba6a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ce597801d403620635b633b079a58a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4645db6444f5f50530c0ce48b0bd46ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 128, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d39013bc9cab525588a16d2385b8e8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4645db6444f5f50530c0ce48b0bd46ac
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf52713f8764c6416d01fb05c81c5368(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 64, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52ad5a516dafd56f9fb681786a18147d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf52713f8764c6416d01fb05c81c5368
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_482ca9ad547e866850e30f951a4c7c9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 32, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_745369ff97ed979b2a357d121b3373de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_482ca9ad547e866850e30f951a4c7c9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_941da0c0cd2787426935c4f47f9f794e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 16, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd7b2118c59e0f0347e52cdbe536a326(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_941da0c0cd2787426935c4f47f9f794e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c59e84fed869c6cc8325bfdf6b63a580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b925cc2cd4a91157b9ce9350fa450fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1da32c099953f123860b23f7399ca5f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb15bb06bb723bd0fe291dd8f819bada
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d4164e35c385aa965573d2d358578a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d629a2b2011078cdbbdae74aac108fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b51069041aa44c5e825dd209ccfacf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c035815ef90c0a6de3b3094728c9cd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aee0eb8f28e8ff218bad3bce63a684a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21ee47002ca433fd2e518897cb08801e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db8531b63edba623cfb0eb2258fb1c40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4b9a8701dd43d50f9740b783464dbae
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60cc23cf3e7d01f8c3d98dc8ff0f6ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3be3d3044c3a08c49766adbc5f4cbf89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3aae4c7005dc63ccdc906e4da5cb4d97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f64d59793e88a91ee6652df5262e6e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cf18f7da918fd5b3813f3b10595aacd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_402af2a501343cc5e114bda69cb2d457(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50fd9b6e4c41df774ac7dd8d93193618(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa455124a9025bcb0af7643ed35fd87c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28cb794c381baf9658d1141320a4b9d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_309714eb861af6b9fb13456f3d9c6483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ceddc956916a7b6f391373fe1743639a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_baadcbebae02e05f449dda8f1994f9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c1e0a16a41c91d317dde4505253ca11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_737aaca9cda73bf89199c15cf042a03e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a17eb8160a041bff246b710fb4c645f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8180beddade27c19a2e7edc810e7c9ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1636c00c0fb2d3c59df6cc205ec54854
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a715f654bdd5232a8b417d9700a8d26b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4b9a8701dd43d50f9740b783464dbae
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6fe9510d9d56f6015a7183cf1e8dd43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4f4d04de0043df773bd3dc3c8293993
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d4164e35c385aa965573d2d358578a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7eb22c867363d19a8870dc3a7fec6d81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_025d654729e7a9c033a53c4dd9186ea1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b51069041aa44c5e825dd209ccfacf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a079ca501124e63f85d1444bba22fe3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8adceba972fd7bb99a77b88936dd8b99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_795099eb19ce3e66ceb068586075608a
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f24037e5c16c3fb5ef0540189ee75e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14c6d0564c1149c8bb15622289c108b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_361c1bc941ebc5845c47282ac94011cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88e5a695a138c5a7029a0b2be8033a9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1880dd9028c02f048b92211fb5dad5f6
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e2fdccac17e2ce0c23ccd837d5145ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27ebb52474a7c58865cf912c53aabf9d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7416d1cf34daedb783deb4ffeafce2d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea17e1de9fab9b6a824617b1a84d2
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4282e2a290b79c088d90063dd7329f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3431ea5cb8d1538b50beaab2dd98a84a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_64fb3b519f8404b4da0ab4ad46671fbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32f505509fa4786a3fbcd34c5c8e655d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4f4d04de0043df773bd3dc3c8293993
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_365307a86b9b90bc805a39c3bfb11c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57836e1df3a60efffb8557662ad31f15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f158b330569caa2ee440e98115b4bda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01943bdadf9b5b5b157adc9acc758e6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f0c55056f30870cf6b53863cc9a5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_844f00d1d36437164c981b1c40ecdd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e94259a364fa16aca57eaab799716fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_124d07456070339deaa50364f6ca635c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2431fb051c6365386ee9fe9adb75a3a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44e93eb7ac25ec3af1bb25bcdfe9adfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb15bb06bb723bd0fe291dd8f819bada
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d5580e7d70faa6cb72e55678b291df9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a723bb00d0989bbc030d92cdb98ae9fe
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a560917b196cf7237e782b2bd294c34d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50e7941ad1f4c3d965da50260dc17822
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6236ac0ea2de78dceff6ef9e5420344(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_025d654729e7a9c033a53c4dd9186ea1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c81582be1288e0b38ec00c1123e96a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1088deef2870e0165fe9548f2a48838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2ce7040eca8cbab9d1d92b0db42fb275
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fff99f236436a9a8a17dd36aa9f45b21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8480007012f4325d31c1a5a3062524d
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2f66fbd4bb00c4a58da695d41c565c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0050a12f0a3e88c9b56608fa9feec33d
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_83a6c42f920de001971cfdefcfb42f01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a02a78343c3ba6a629c90998775d683
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b3cb53cff499078dad35244250db7fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bcfcacb92cc760e3a7ff211c2f79565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dc7761f5b4b7bb2e19a9e828e8990f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93960a58176e24a7137e343020853978(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b925cc2cd4a91157b9ce9350fa450fc
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_178d26934d3d2a8ea1cc4e47963a7abe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36180af9cf18b32f66f49f2a46fa317d
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac1517476ab269d8ae75057877516369(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da94eb11545b92d4703e28231f737ee0
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_858b3f6c787d9b69f4b6437244951bb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_504f2a430bb56c8b8b258c100c36aca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d131046c72e941483437a0faa1a4c35
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db21b58b36265d59be1e4396210501f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1168b0286f1541cfe1954528879997f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acc5f78ffdacb9b9711c3a4545d8be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccd252dd17cb6416f44b55305b80111c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da94eb11545b92d4703e28231f737ee0
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c2998e2a920afa6250fa8ddc1d17a47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d69a6103d936090bdb66db41b198aa7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_025d654729e7a9c033a53c4dd9186ea1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_599b93e1a20848dd7133e4979a29a4ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb15bb06bb723bd0fe291dd8f819bada
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f4dca4146e3c7b8d015893aaf2dbf01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14c6d0564c1149c8bb15622289c108b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2db4b97d4b10db81b910927c287bcdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_633c0e2fbc6db95c98e926bf2c2bf903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_025d654729e7a9c033a53c4dd9186ea1
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ce9f45caead8eba523db20a6c582287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e12635fab2a725f76cf7890835c8f8c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82ea6b6eee7e1e601a3e33370657098a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4a024f7af5ab243f1481612a4c07b05
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1d51bb3093a880e402ed541953d116e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5612612bb646b1c37cc22853b6ed94cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb9dc5e2108e72fc7f44b9ddf3201785(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba1f2bd9a94920dc4c7a2af6d7d31554(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67e683b21061e58b558804a6bdb3ab54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2359b2ab7326afb9cb6939cb33fae1b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e626736f7189803d672bf24eb89903
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51ff404a2f02f50c1981217f72159ee5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e626736f7189803d672bf24eb89903
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cda39061e28b4d683844520a3cfb532(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e626736f7189803d672bf24eb89903
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1315d87b2161c6d1abf7ac08a52b6c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e626736f7189803d672bf24eb89903
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_399d3258c119c3b93a1f4f36af365f06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_50e7941ad1f4c3d965da50260dc17822
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_763c4002f70df1171b0db9c505ca4f5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65e7882c27d74ef5d27a343485f0edd5
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_622f62ff4d2ba8e42253df084b1794eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3229ccb3029a91a735aa5f5bfab6ef52
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39ede9a8db17cbde20245b04ab0db1f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ebb414d7953159cbb789766562992de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e86c911cbb9a19d7cc4aedca5231116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29cc8888b8a6597ea31bed8fab1f07d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_058249e6bc22556926eeecc86f9e4dab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1671bc86c24f9f0309603e021ff1ac2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54deb7c7908b56e2e91c264cc0b13b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9f113f3fe3e2834d2773489f65d5e54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22f7da8a5ea87660051db52c6217d812(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05236d60b23f25bb12664d119d161fc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f600e68bdd4e21295362cbeb064f555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6255daa051e91b7951117763166b81af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f24900b11f55f34ed87c301f444120
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72c51544a33bb388386ee66aab19a573(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c33be5521df8badff101415bcb88a6d
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecbb4bc1206f553b31c81ca4ddf4c4f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ae2c939cf103bf1dd05273b097b3036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96fa52cddfdfb197dcd6cea697c05fcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 92, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac4dd425df2ef94671799a0a9a7759fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cc5451a3cb06175857659337311712b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebca71a2b6be1ba735a0ac0e38938d06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39ee9c49e62f79b77ebf7406d500db8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3665f2806f8b13c76c77173649f28d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eff1d25f3b63910b116c506e15256ef5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10873e2c7dd0662813a992606a4faea2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5582ee8dcef92d93aa7a9115386b9f5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28bc57ca7e07569e428c5c18bb2dc51a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea91e980738718e157b91369008a2cf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af8d935f9681f6410c76d55b4c354387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c222526be7268b406933d603fbff9057(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be325627cd55400186660f9cabecd853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a6246a198c8287973e68b3ceda24fe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_351dbcf3224b74da366523a13052ef1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_866b4137c8f6b614679f4e9e83d279f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c21734409ca0a9d297420a65abfffe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c6348a86ea2d64041b7e751598a345b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 60, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_534ca119fc877c6211290818f2617182(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfa9bdd884d0ff8717b4d62256b9941f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad1877fe708385cc189a0d4df3efc509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44fdfe9be5123fb3e8e4eab6916390af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9efee9da50ea168547b966a9ea816d51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af7287320b3e04f046f0d398573fef26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8da12591224934238b754b7af318f19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa4feb011790a997b3902d564c8ea4de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6936454ea210692f42c3dd1226c0543(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e35f093a06fcd242eaa2e8cbb2f6c1be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_134c2f958497dff185e0ac636ad0019e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a85571244114b9ab6cf6926d01556346(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abea680ad996fa26222a5ba1846bcd11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccca2fcf662904f33842fe4097df213d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aafcf76ced1a0ece5055e6a3df448831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e0738214e415b707a93de1048301e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_144814c397b00369a888ba8f37716741(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57e40cd0debe86d3d03ed6fa0b4860dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1de3f36a94cc6c2a752131546490efab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 1024, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_246f5290018ba323eea342f30f2d5b72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1bb75dbc4067ee1009a94e0056aedee0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62c4a5fc79a5ea515224f68b83724cfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_178171e092d61bfbed73cd38ca277d6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_513bdd30e8ee005e543560887388c474(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f5656cdceed1a57d07ba1934fe3866f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d22348f0e6b07449ac7919497b03f22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f077a8c9307d63938b02d88cd479e47d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcb175a35f23da7400f46656efba6a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c934870ef5a33e81d76f8235f025c07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e326024d4638b71fe1319d583a48088d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf95b1323abc25d63dfc4730afc07ccc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5146c72c64a75c9bf81bf26461a87c25
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3912524e9d06ecced673341957aad496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0256cd40851a0c0f8cf8a99854b42005(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4d46110686466903cf24c07f354fa71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81dc63d9ae1942666fb3004159f178c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1321dc2a08da864641ca246e2a34ecd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f4ab66901df3f31bbeeed5b1045e5dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5146c72c64a75c9bf81bf26461a87c25
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82d81b08a48e737c919d54748631a0e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e326024d4638b71fe1319d583a48088d
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41d5cf3fc6cdd53596663b79a887bd3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcb175a35f23da7400f46656efba6a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_087914b58e74f540d7155e4b2fcc1920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03c540c13f4c8a62e6fbd12c806b50f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5146c72c64a75c9bf81bf26461a87c25
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_35b5e968fe4b26375ee1c9113750ca24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e326024d4638b71fe1319d583a48088d
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93caf0eb0998531ca496255a1c0d145a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcb175a35f23da7400f46656efba6a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cd553d7ceabd7f6b4b7e5c57dad9f80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_629fbe168cea177c87df95c74b02978e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_800dc3d644cb4513585fa6b9e9ee7948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39ee9c49e62f79b77ebf7406d500db8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68f7722a5778e021162d5947585d8417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e55ce7518dfdd4f5cc6d61df9251f34e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd85f039bb10ff502acc746a8f345bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a81c821e8e68583a2612070315c64b1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07969d7e81a54dba7e46f0fa3c629b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3da5840179760205061325b26cace3ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d6bde30fe47b66776d096fa7776f8d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b2a7f98402b5f342487b23686490c32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33346bb59fb104cf3b0bbd59c13b8301(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a9512cfd9326ad56f25628b10b6a16a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8822ae65c00cebcf5ad673f6b7a8ebe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fa426c1b4d75b13e26ff84594dd214a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_647f64847ee964a8b4d495afc30dff1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_549e3f199dc4ce5af2a4c6628111db9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a5bffa3d0000f796a1d9d996137790(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ba7f1f70af3bda16d91daa5df0233f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef34d2847d28ddf48437d7168413a7f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45762f786fc2311a4c9e3269dadbab6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cc7967334ab1aaa1b0b4f82315c0804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1843f5edb49c2b1ceb5b6e16c043e78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4801736a792c2e81990dcc56633379e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5551867c27eaf7b8d4d4dbfa2d03b0fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c73387b024d26d4480319eb26fa82f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd8afef19556a246b1dfc55b3f8d5370(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47705e3d62b64565a8c5f19d2e654889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22ca8b97f81b9f145b501c0384d39cf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db1eb740f6bae2991b6a3e1a87fe4c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae1e9a7fed6796a735fa0431825f4e8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6165c68500f3ff8222e5207922a54f6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df1b21a9d7d0c132774d25cbae91df01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 320, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95f73c3f0cdd68cd6d1c1ac51b42b0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], False, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f559d30a60a014834885dd733234666c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1683e6e48c172b4247c85b96cd77810(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([43, 256, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26156c0a0770c1eefb994fc29a309e6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e5aadbb303be8d7d26a2eecfd42b0b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_818914026848c8f4dd41cbb2311da306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ff50d68a552c9ac9fbf92c921e9eef8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d50cbdd1363624833f35227891391278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 336, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bc346c05b9aac058a5e05c918a5489b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43c9ed569cc6ac5de4d37e5f4dada9be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d26c86536063d6c14cbf32e438775d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22e84ada86eb5420d011b56a4325e598(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08bf7dc410899192afb2ce83ff604433(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0c378e924073a2cc5a0ac4458a8c395(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9b3e9c6494a595a19021062cbd3cdb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39a3125508c27c5e3131979ad7570379(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 36, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44fdfe9be5123fb3e8e4eab6916390af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cb422e31b0eba88f1452341106222eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98d8fcc6f4bba02954f0c5b1440a50fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([10, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53e5dfabffec08746473b61b3a2d50c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_066792b4e9e88d481059a560678616cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb38826e4133b037acd5a64fcdb42cf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f3baa90f6de1192845ec833c45932b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6da0e8529c2fc1d8bab0b1bdfda62d58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38f7345605a7df26a84c5dbc152250f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63c833bd2146802159b776e64a09e186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8cb0d6d69e452cfd100815c8d825e61f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [7, 7]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_819df0ce4c13201005130999b124b5fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb0d6d69e452cfd100815c8d825e61f
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_819df0ce4c13201005130999b124b5fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb0d6d69e452cfd100815c8d825e61f
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e32cde3d08b48736e9213400c3a2580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb0d6d69e452cfd100815c8d825e61f
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e32cde3d08b48736e9213400c3a2580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb0d6d69e452cfd100815c8d825e61f
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f4fc27fdeb1c3d07db1966344b90753(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb0d6d69e452cfd100815c8d825e61f
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f4fc27fdeb1c3d07db1966344b90753(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb0d6d69e452cfd100815c8d825e61f
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e2102e47df7d3b3119ef8672ff627b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb0d6d69e452cfd100815c8d825e61f
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e2102e47df7d3b3119ef8672ff627b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cb0d6d69e452cfd100815c8d825e61f
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_333ea654d7ca65fe0384bcd61e1cdb3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [28, 28]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cfbf2003eae6035894f9fc7d1ab7d062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfbf2003eae6035894f9fc7d1ab7d062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36ef66d23e39cabe81cec5d53d1b6987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36ef66d23e39cabe81cec5d53d1b6987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8a8aa51d0131e24470e6b67e88be38c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8a8aa51d0131e24470e6b67e88be38c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5cf9d1ef4fd53fa25f2ae56999f1320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5cf9d1ef4fd53fa25f2ae56999f1320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90469da5d2040aaded843b467d1def9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 480, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a01c1072c92259355d8e3ea5e8015d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22ca8b97f81b9f145b501c0384d39cf9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab367c591716a202d79e191933b90b4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_242ab21623c9147d29d04bf8cbd2f3b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7b2b7941ea4e3a202890f56459c9753(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ea39a4896348a1de14b1632616f0999(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27070c35c8e359c551ec0f217bd04156(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5e5e5182682f3a1bb4af95aa95edff5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba2330628a67e2aa731524c76c3103a8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [14, 14]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [0, 0], False, True, 'NCHW', 'avg', False, True, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32c5ad1475e6fb39ea83479ba3d72342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba2330628a67e2aa731524c76c3103a8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32c5ad1475e6fb39ea83479ba3d72342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba2330628a67e2aa731524c76c3103a8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9367d7256d8ff6da6703fb46c0bd761a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba2330628a67e2aa731524c76c3103a8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9367d7256d8ff6da6703fb46c0bd761a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba2330628a67e2aa731524c76c3103a8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7c9c58e436dd3cddf4e03eae994e00f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba2330628a67e2aa731524c76c3103a8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7c9c58e436dd3cddf4e03eae994e00f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba2330628a67e2aa731524c76c3103a8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ff75fe8d57770a86d22f40b543e20ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba2330628a67e2aa731524c76c3103a8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ff75fe8d57770a86d22f40b543e20ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba2330628a67e2aa731524c76c3103a8
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b24e9b259ea95a5479b2cd026ba4cc06(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 2]
            return paddle._C_ops.pool2d(input_0, input_1, [2, 2], [0, 0], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c50a43b48e8e34b0af7ca01f6d395220(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24e9b259ea95a5479b2cd026ba4cc06
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 300, 300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a7526df5c8b4f559f9aba81bc32bbad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24e9b259ea95a5479b2cd026ba4cc06
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 150, 150], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f96f6bd90f0d272970b0ae4c64ed4528(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24e9b259ea95a5479b2cd026ba4cc06
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 75, 75], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e93eabcf7c74fe909d1fa1ba472125cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24e9b259ea95a5479b2cd026ba4cc06
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81eee03d16784789ff52c2a8ef0bf6a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [3, 3]
            return paddle._C_ops.pool2d(input_0, input_1, [1, 1], [1, 1], True, True, 'NCHW', 'max', False, False, 'EXPLICIT')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b50485cbe6c0225b3f342a6bcb53dc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81eee03d16784789ff52c2a8ef0bf6a4
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a27ce69f4eec6b4cd6b66046bb15c765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ae24cf144bb429afdcfee06ac3a8317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfd9f6b2668ff049202394b8f1b3938b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 60, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b09cfaa8a254e2e3ecd915ea7bf06133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c33b29fc9b706386d9a70fe227546ce7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcb175a35f23da7400f46656efba6a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab18576067af769327d95ce5a549df78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e326024d4638b71fe1319d583a48088d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33f1a27e81d1a53b2fc07f8db19c4047(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5146c72c64a75c9bf81bf26461a87c25
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38a0ac9253366bc0469f68ddb2e56219(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da4ec050f29cc73fdb99ad0a74a6d16a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b393d38763d2d9cbba694a8e8489638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_819a157fe137fe532a8f1444d0b9ce45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_762be137b1258ef230a2c86671e6057c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7291d7fbd3737132da715adba7afde4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49726681b862f2ed4f599372a7ed18d5
        def get_inputs(self):
            return [
                paddle.uniform([22, 7, 7, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_817ce72b6a1958c84ff5b0f226fbf3eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa3e8fc6a5672ba7e9fc2f3e501b251a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f25b5708f0bd139cb07fec5924a5284f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d8ec158c441aabd4901fb80c02863bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dfb4c877c20ba80a4bc9b62cf978fa3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3c7aee812d2a38a4db14967b6f885f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be076c3d3020660e1e995aa1b213e121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44accfd304df6315a27bad63ff2436f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_333ea654d7ca65fe0384bcd61e1cdb3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd1ae884c99d38ce83d77ee5738a86d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_875a4b95ed50f14a4364ae8e426724ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2f74273dfd44c78f9cb4676a5913b42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5339fd88a577d1773bd12cab10135c2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71c0ba2d47b69ead35ae8fb3ec0894e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 16, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7400bf86bfda1fb2c1e764bcb37b42bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5cfcac9b7b5c6745bf91f5e76fb6e6bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f47192ee2a30c09efd1da2c358b62bc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_783fa8e44e18132bde4656825bcc3cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e740fb4479fe9ecfe47abebee18440f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([11, 256, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77765baa597c7e0d3e38e7f75e244daa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df430f84195a456558b7bdd4450a0ab5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f1b8406756d0356ed73c1a2e689c411(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee6b8412a9996085e1bd7db0496a9ec1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a5f6005d15a5344cd217aec8c10ae6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 60, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34de2743375fb3bccff60f90bc1f1327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59de7179a611a4c3dbef82f322f52e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09e26f665abde78416941b00b45a7d82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_421344d4187a0cd6db650369c5fd5b84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08199baf6055abc15598f5b63f688222(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86265dbcc230fe543fec14011f3f032a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 336, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42fb8353224e6973b0cb3c092be7e19d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc55e5a03b08523fbc110d37ecf919b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e41c4d355cc5027d0ecd04d0e1b53ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_937da3af0d78d866884b4cf1e9e76853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6390848d7fa37f90aea3c684f83c9e6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af94c6f977d64194b5715d2bed3317c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 38, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adfae89d34c09456e83986e018613310(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bf773429da85e0e35a2bbd5e0911d7b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fee2aa63bd0f5b14e95b1982fdc53ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75cfe6fedaf8ff7d32ce0360b75c1f7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c361ea030493dce89f475571a94ccc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3912b9263480b38232074597240d1b95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daf631cc05e557b406e60e91e34a631b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f021d38aebcccbc1f7c97c4c59368cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d6b75bf600dd1d0a5591b56086bce98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 320, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2386e00695dcbf2261a9204a15bca6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a78fa06bdebd3422f1f0128ebffceb32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40738f6a3220cc546522a1d7d64a93b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e498774a5073da29d57ba00fb77f20d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e773072453e991dd75d7688f8565494d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ae1102d8d2bde94acfa11ff16939424(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 44, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f32edf88d6987a499eca00371386072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6797a36a25decf40f79536a6c270bcde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 1024, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cfcf9ae92ede7f75d27bc27addabc6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b95152eb202c86ecaf682cf21fef464c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_421344d4187a0cd6db650369c5fd5b84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0a41c66d4fb05740fc8648cdf922d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bf773429da85e0e35a2bbd5e0911d7b
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bff3c471099db32ec97daf9f7c3f559f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b71eb07f61d479a370682a8a79db76f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b508225499c5b4dae08749f94f98680(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22e84ada86eb5420d011b56a4325e598(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8c74dde4dd64385e48049b53d318a26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55dde47ef0b66cc424b4c39177448523(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e1931f4a2ba3b833d13b5757d63329b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31f63431f497c14cf12ccfdbb35a2174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 240, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f90b2df96ac8e165e3d9bd0e4ac80113(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f4c3c399f7b8b6be85cc81b52b3573b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ecf55562d6b054f61b925f0174788cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc58575da660a797b064cc2d6cf89a4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 240, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_242ab21623c9147d29d04bf8cbd2f3b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26bf897ffbe9c535e10c15874af77a82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5339fd88a577d1773bd12cab10135c2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d08db7e80c57e22d7d8c7c03a9a7323(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e688b057fdce1df47acf2d18535f1246(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2efa8daa53906d4daed9b1dc316a7e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f29f805bd6da530ac3e3c4d3ec95e19d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b02efb3bf52b210acc39c417c7038a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63e875e090bbfd61a6eee2a4c64d1b3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62a289d1f11377fb88d01cb546ddb585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2257c4798e31ccee4800c6d2b45675e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03b29678a1b7e2e2c4786f911f6bb217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ccf36cf6afc5472652a0bb88b923b60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4423ef5f889e37e4fd1734af3add9e2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_534ca119fc877c6211290818f2617182(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bbe30f7d9816168cb0eadb1284cc317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29cc8888b8a6597ea31bed8fab1f07d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c21734409ca0a9d297420a65abfffe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09d74a68eea1fca5ae9d68398c373922(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47be0b6a13923a8c282a333020e65379(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e02278a8c598994aed83f90ef6ae6a5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc22fc6f02d137c2fee321dc3a35df68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 480, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c65f350b6c01a40597bfde63db77d435(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 36, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cae7562fdb12d93f8654cabc39314a6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cae7562fdb12d93f8654cabc39314a6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7215dca8dee76b8300f7c4e677fdf80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7215dca8dee76b8300f7c4e677fdf80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b59d56ebd9505f78dd551480a72fad6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b59d56ebd9505f78dd551480a72fad6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8566eb427d1c60dec17fcc5185574875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8566eb427d1c60dec17fcc5185574875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5bc3662c427a4a95f775a7d19b0e3c4
        def get_inputs(self):
            return [
                paddle.uniform([10, 1, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f028fed5b4ab072eb68bce00ef8df45e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_407412ae875f225712e029c60f86265a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bcd3258996c4f88b7b55880a54f841e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f511badd00a66784a0ca07f2a1e34ea4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_341f8986821ed9eccf10d581ad82151c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fca81d5c59a2b19298e4d79b9d9698c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2629c39003f3c8703643fc53d26347d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cb69282d09833b5ffdeba56919e8976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a36bf7166fc826e8a8165aeb3135e2ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 156, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_929e9eb7e5ee31f2f38370e29cd401d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5146c72c64a75c9bf81bf26461a87c25
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e966c5edb1ec242b0858537b4f9ae6f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e326024d4638b71fe1319d583a48088d
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffdc5ad1af9fe0f0c6992f01d16fe8e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dcb175a35f23da7400f46656efba6a2
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ce597801d403620635b633b079a58a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2659142df9490b914751e078ad7a6c0e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be325627cd55400186660f9cabecd853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a6246a198c8287973e68b3ceda24fe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9283e2539bd0c7d8681bd156dd9a9dd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7458d84d421f52a242a00ac2c8c5b8d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da1a238a26c61268ae12d2b3723dc565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b3d43fd7d3cd8eae954c79c4ed8eb44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_653ab9b114d4ab63a6b15088f1774cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9cf75f2ce73ec9ee6d0bc31b3293a43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_303bd08c315b3c8a9233c70e43d19a1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c035815ef90c0a6de3b3094728c9cd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 872, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41ff6c6a8f7a875320760bc3241ad6ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21ee47002ca433fd2e518897cb08801e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5453e876e9946b0fd771c10fd9db6cf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_421344d4187a0cd6db650369c5fd5b84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3be3d3044c3a08c49766adbc5f4cbf89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([145, 480, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3aae4c7005dc63ccdc906e4da5cb4d97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([171, 36, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_476b1a06647724c96d5c0d0f1461e65e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cf18f7da918fd5b3813f3b10595aacd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_402af2a501343cc5e114bda69cb2d457(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50fd9b6e4c41df774ac7dd8d93193618(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b951516e08adc9a23fabd5a45bedc32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_309714eb861af6b9fb13456f3d9c6483(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7c5fe5bcb58256b3a555dace8e69baa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e2a3037cf564ad40f2d3d142ff032e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_27845c8b4c8e84ef4470bbe87be05e34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_737aaca9cda73bf89199c15cf042a03e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_420123ba21bb6d2a30bd5f2719171382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be325627cd55400186660f9cabecd853(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a6246a198c8287973e68b3ceda24fe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9283e2539bd0c7d8681bd156dd9a9dd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_653ab9b114d4ab63a6b15088f1774cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f300d14ea02e1437157c0290bae75291(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_303bd08c315b3c8a9233c70e43d19a1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40738f6a3220cc546522a1d7d64a93b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_505c00d078c4936b22f3cbe6116decdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5d072df85888d3548b36acef3606408(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4492ff4ed8baea857ec0ffe2a8fe1221(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a97ce65fbc8f01f57a7c5e6be91e3ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 336, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff12e04bdad459756fb05340b86ed597(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31996cd524d619eebf0a9b057a788f71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 56, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0b0846bfddc8304460c90c8527aae58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec5799a2eac072ef2d52e7d3dc0be182(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_427ac044655c531841e606c6b4defc74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bec2fba1d5099fd9d37a22aa63801e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ffb121243fd949647f997929b3872f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6936454ea210692f42c3dd1226c0543
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f158b330569caa2ee440e98115b4bda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01943bdadf9b5b5b157adc9acc758e6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f0c55056f30870cf6b53863cc9a5056(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_844f00d1d36437164c981b1c40ecdd20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e94259a364fa16aca57eaab799716fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_124d07456070339deaa50364f6ca635c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5e5e5182682f3a1bb4af95aa95edff5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74ea86ea7ba869a4a563fb5e0c873c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f0ccf6cb6b84de6048d0311ddac4e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1be61b7dca32bbf3fd00063ff37f407(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_491f10a1dcfc4917baae2082e25e0aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_498b51d0ec0768438a22356674238701(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01cacb79fc874386ccadd2c4d0152647(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 109, 109], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_def78861d6a78dd1b0d7c557840b15cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 54, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_389015e24676a6f8ba79a534b44bda59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06808a307623cb9bbaedccf38c6bdf0f
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6c8a4b43401c0e9c623dc0fa10231cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b3cb53cff499078dad35244250db7fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bcfcacb92cc760e3a7ff211c2f79565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dc7761f5b4b7bb2e19a9e828e8990f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7247049488065b343a7e0b1fc85c2161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 200, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20508324b6c76c938e0c40863526d27e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 400, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a78fa06bdebd3422f1f0128ebffceb32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05dff6ecc32506879aca37818975c6fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_641aeb4653d45dac6d8fbc9565615ccd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd8afef19556a246b1dfc55b3f8d5370(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84141898cf8c3be09a6d2eee0b8af6b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7b2b7941ea4e3a202890f56459c9753(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e55ce7518dfdd4f5cc6d61df9251f34e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58a97bc24ce781b2bf7ad3e15f550169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b814f32b09c4780c10ea60d8755facf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef38fcd195c58c79a6328382a50d8c96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42fb8353224e6973b0cb3c092be7e19d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_088ce15e42ae752c161932f0e65f2070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06ee1e753bd3337054a259f29b4ca6cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_407412ae875f225712e029c60f86265a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1d51bb3093a880e402ed541953d116e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b89e0e70b93047fa068c1d4f5e03843(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77fea890bf8bf5a5b477429355c7d1d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a28bee5a8049361d8f16098736e0a541(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c40ab29dbcf533096f8a0812e3bebcf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 328], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c82ddf86eac948f06242f693b06c77a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ac78246656fe5920d7a4c187a0c20a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c26aad8c5291b611134a12e57da1b66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91d03c2e7d6037004df1e73449507f0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 184, 328], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1654148ca97caa55f904b31ae77dedab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb7bf72edebf096672dff2deb05b6d13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63c833bd2146802159b776e64a09e186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c18ee98ce65e48d1edb826981636a1d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e86c911cbb9a19d7cc4aedca5231116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29cc8888b8a6597ea31bed8fab1f07d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_058249e6bc22556926eeecc86f9e4dab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1671bc86c24f9f0309603e021ff1ac2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54deb7c7908b56e2e91c264cc0b13b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9f113f3fe3e2834d2773489f65d5e54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b43cdc0c3f8c0262e6f800760679c79d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22f7da8a5ea87660051db52c6217d812(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f30f70b5785796993b6a405b63a27f3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05236d60b23f25bb12664d119d161fc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d536db09f4dec98d6533b073390b31c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f600e68bdd4e21295362cbeb064f555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 624, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9efee9da50ea168547b966a9ea816d51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ad7e29b7e2d3d27228ded6457a7d525(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0f32d512811e3d805f76efed5e2d765
        def get_inputs(self):
            return [
                paddle.uniform([1, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()