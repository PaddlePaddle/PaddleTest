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
    class PrimitiveOp_c1615919e73c762620437ff2a6c807fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 1, 1, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7265f0e5209c6af9364e4f94bea20b7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1615919e73c762620437ff2a6c807fd
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 1, 1, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_9d32ebbf8d7e2568153eeae8b6a88bc6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3, 3, 3, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be842945963961c0499073183937d21f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d32ebbf8d7e2568153eeae8b6a88bc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3, 3, 3, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_fa8ad9d0794e46c51b93ebea42c2e148(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2, 3, 4, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 97, 97], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42972a44778709bd58d4175d0a5cdfd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa8ad9d0794e46c51b93ebea42c2e148
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 3, 4, 0, 0], dtype='int32').reshape([6]),
            ]


    
    class PrimitiveOp_99011f913eeb85e738e01c999098d868(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2, 2, 2, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59e4d418f4788cef8d8fdc3517692d31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99011f913eeb85e738e01c999098d868
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 2, 2, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_566195a1293a1e8bd1f1cbba32da319c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0, 0, 0, 0, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 1, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2de4553221671a0c6adc4d0bef0d3a34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_566195a1293a1e8bd1f1cbba32da319c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 0, 0, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_a2f1e22382e9a2a1f06bc2676931d03e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 1, 1, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37185a6913265dc16fb94fda6077367e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2f1e22382e9a2a1f06bc2676931d03e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 1, 1, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_071a9a04d6ddf51a6b83117ce5b23b92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3, 3, 3, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87f3dfa590447af26e93be7e099c0d1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_071a9a04d6ddf51a6b83117ce5b23b92
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3, 3, 3, 0, 0], dtype='int64').reshape([6]),
            ]


    class TestPrimitiveOp_42972a44778709bd58d4175d0a5cdfd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa8ad9d0794e46c51b93ebea42c2e148
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 3, 4, 0, 0], dtype='int32').reshape([6]),
            ]


    
    class PrimitiveOp_6b168069c737744679c3378b5104d73b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2, 2, 2, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf7da2070d4592bb5490abc54533ec9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6b168069c737744679c3378b5104d73b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 2, 2, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_4574feacef7801dcda85820566aa91bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0, 0, 0, 0, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1, 64, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2fdad06aa3c4d5623c0a34aaeeabbf93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4574feacef7801dcda85820566aa91bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 0, 0, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_a0e78a8c7e884aaaf73b0a20a05af1bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 1, 1, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6efa39db53e9cc307b8f9ed4ca9155a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0e78a8c7e884aaaf73b0a20a05af1bd
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 1, 1, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_7adec2950b6b14a4dd25fa24497ad13d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3, 3, 3, 3, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_040a079740ed08fb0f2fcbc95a5a9c07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7adec2950b6b14a4dd25fa24497ad13d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3, 3, 3, 3, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_68bdd7013b8b476ea3259fd52f76268b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2, 3, 4, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2842b1ae3401c66321b7d5de6f5bb7af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68bdd7013b8b476ea3259fd52f76268b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 97, 97], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 3, 4, 0, 0], dtype='int32').reshape([6]),
            ]


    
    class PrimitiveOp_06c2d0e8f4cbbef95c11c4cbbe9b695c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 2, 2, 2, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7f61455f4f6742f77226967373a73cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06c2d0e8f4cbbef95c11c4cbbe9b695c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 2, 2, 2, 0, 0], dtype='int64').reshape([6]),
            ]


    
    class PrimitiveOp_c2f2d5d932e45308ed2c94c7509f9e69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [0, 0, 0, 0, 0, 0]
            return paddle._C_ops.pad3d(input_0, input_1, 'constant', 0, 'NCDHW')

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2b01d8e8ffd9ce1616148a1e6dfe52a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2f2d5d932e45308ed2c94c7509f9e69
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0, 0, 0, 0, 0, 0], dtype='int64').reshape([6]),
            ]


    

if __name__ == '__main__':
    unittest.main()