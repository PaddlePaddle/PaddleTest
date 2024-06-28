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
    class PrimitiveOp_c13c07819324140bd151d939d6b61a18(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29ffc1b820a7df0ef0246517d1c3250f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c13c07819324140bd151d939d6b61a18
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2db490ae667f38bd6d576336953f314b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76b3a76e882cd8e98256a021949d2aa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2db490ae667f38bd6d576336953f314b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_09d4845a4d9dd963811e5ee784b5407a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc583b51546d64afdfff7668fa2e9c57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09d4845a4d9dd963811e5ee784b5407a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f1f2a5dbdb8d06b7586bdf9b8ea974b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15787d7da144359d2d4aa70981f57ab3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1f2a5dbdb8d06b7586bdf9b8ea974b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_01d1954dfc87a32b602cd9694037371c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_065bb2dfe07bded804b2fa2427c65f9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_4a4ac308c465319518eb8c55c193cda2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_538c7d5a16b007b404dfd266ad839df5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0a26495d0104a144ca25e1dcf643cb4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b1aeb3d22f0fc1db76b0a23992c93c4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_02e47bf781c1e34fcde4e2a047af9310(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 168, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_057d815fdd35e5589b45dedf7ad356a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02e47bf781c1e34fcde4e2a047af9310
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.173097625374794, 0.24690791964530945, 0.12752406299114227, 0.21726328134536743], [0.42141810059547424, 0.32032057642936707, 0.22069108486175537, 0.25408151745796204]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_72bcb72518b3a3c13e2cb74526f9d3de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 84, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c1c4c1b98d402f6eb737071c4ebce34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72bcb72518b3a3c13e2cb74526f9d3de
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_0032d2f6cb20eed35a9fab273c1873ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 42, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c09fecf72f68557437751e30f90b314d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0032d2f6cb20eed35a9fab273c1873ab
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2afde830413e236c4254a42947c4611d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 21, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f14dd180dd904e8cffc0f9309e2e7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2afde830413e236c4254a42947c4611d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_eeba2a01ac69d8124e958e02e9b91ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c13c07819324140bd151d939d6b61a18
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a417d93dc6f2a12cac79354d2407c583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2db490ae667f38bd6d576336953f314b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_07bfb4b440fdc1b489ddc89bd5a1575a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09d4845a4d9dd963811e5ee784b5407a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_50eadf98b7f32ef6b46124604f3f0e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1f2a5dbdb8d06b7586bdf9b8ea974b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a6cacad2cfe67b93bd01295269a0a5c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4688277244567871, 0.30018413066864014, 0.23425810039043427, 0.3811199963092804], [0.4602409303188324, 0.1876598298549652, 0.14904391765594482, 0.10620903223752975]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a09c6e7680ad49aea7bdaa91a07b810c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c394dc69b71dc4a9f543e97796719b01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_861cb12234458f04eba653522bb5daa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_915ab8504ebc67fd479b8e9c5bf87d67(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 200, 304], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_50eb0ee582bf874b47235daf9b6d3a45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915ab8504ebc67fd479b8e9c5bf87d67
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11911441385746002, 0.12901979684829712, 0.18489500880241394, 0.019981015473604202], [0.04435403645038605, 0.22364260256290436, 0.20171409845352173, 0.3678695559501648]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_9c77d774a9d2d8f2362c67118127e0c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbc65648794c364e70a8e987395f8aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c77d774a9d2d8f2362c67118127e0c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_0661906600c858f062420651c2d1ad8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a92b7c24960a12530623df156c030653(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0661906600c858f062420651c2d1ad8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_c44c093ea4b212161162e0190b52df54(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a96ddc47cadb3720c6a0013457bddb93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c44c093ea4b212161162e0190b52df54
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_065bb2dfe07bded804b2fa2427c65f9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_538c7d5a16b007b404dfd266ad839df5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0a26495d0104a144ca25e1dcf643cb4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b1aeb3d22f0fc1db76b0a23992c93c4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_31ff989703af1090e69705ef3fb5ea83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.23170973360538483, 0.44410353899002075, 0.197391077876091, 0.09819187223911285], [0.21788160502910614, 0.20119178295135498, 0.13865119218826294, 0.3262018859386444], [0.0408363938331604, 0.09758158028125763, 0.4664136469364166, 0.2251659631729126], [0.04279877617955208, 0.16735155880451202, 0.17869096994400024, 0.45279648900032043], [0.016807518899440765, 0.10631626099348068, 0.23214693367481232, 0.3479967713356018], [0.2931079566478729, 0.11825201660394669, 0.17887479066848755, 0.19634494185447693], [0.3446366786956787, 0.0214198250323534, 0.029282689094543457, 0.4609162211418152]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_56d2c6c50a6b8931450946a49d444a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_aa24a619441e29b70c993a866e6aec16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_be5935a4cf26121fcae7b5b53e3db9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9258da33cbca1a6cd5a3dcbee8d98b06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.17230640351772308, 0.4015914499759674, 0.22270041704177856, 0.4950568675994873], [0.014866560697555542, 0.2018941342830658, 0.02957601472735405, 0.3901946544647217], [0.46414849162101746, 0.42340996861457825, 0.007422108668833971, 0.24007540941238403], [0.3502897024154663, 0.4091551899909973, 0.3867229223251343, 0.4222806394100189], [0.2405211180448532, 0.34306421875953674, 0.013049566186964512, 0.28451433777809143], [0.4376644194126129, 0.09262708574533463, 0.41278737783432007, 0.17460638284683228]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e1840790a7b1403d2b9286416fabe337(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5266408d73ccc0697ea498ac8bd36305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1840790a7b1403d2b9286416fabe337
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_afa241997c3bfb754dd9737513cfd22b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb2d04e236c7cb10dde89bc77e1c3eb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afa241997c3bfb754dd9737513cfd22b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d7581e8216c5be5f67dcdd52a446b8a6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3769500176b4dd1989ddd054c69b8c7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7581e8216c5be5f67dcdd52a446b8a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_31ed7b22097656c57d1271b6a804212e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.168807253241539, 0.24565082788467407, 0.4337322413921356, 0.09575147181749344], [0.4107934534549713, 0.29058510065078735, 0.46093037724494934, 0.3507699966430664], [0.0019927481189370155, 0.0774812400341034, 0.02200215682387352, 0.1899755746126175]], dtype='float32').reshape([3, 4]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_07c8995f9bb186ed1754ad40687a82cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1a1002bac2103383187cbb1d8bd7ec83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8324004c38a1a3ca30681187bbc1eb59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_547981ff339cdb880794aebc472f8707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915ab8504ebc67fd479b8e9c5bf87d67
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.12799687683582306, 0.006959439255297184, 0.43831005692481995, 0.14842861890792847], [0.18375302851200104, 0.45857229828834534, 0.2913435101509094, 0.38440775871276855]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_dbc65648794c364e70a8e987395f8aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c77d774a9d2d8f2362c67118127e0c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a92b7c24960a12530623df156c030653(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0661906600c858f062420651c2d1ad8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a96ddc47cadb3720c6a0013457bddb93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c44c093ea4b212161162e0190b52df54
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_558c9cd7422dbd38fd07987c659422bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.011920098215341568, 0.4075399935245514, 0.2567863166332245, 0.11302003264427185]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_66fc1a02415f1d43ed44a9fc9bfedc94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1840790a7b1403d2b9286416fabe337
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_24b320a8b568be9bbfa55e68974931a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afa241997c3bfb754dd9737513cfd22b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bac53e74653b2022a2f7ed625cb90cd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7581e8216c5be5f67dcdd52a446b8a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_51fe1c85c657a46cc29d4544e7167b81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.03101678565144539, 0.274365097284317, 0.00527990935370326, 0.19069573283195496], [0.058812227100133896, 0.1637631058692932, 0.025367382913827896, 0.07655315101146698], [0.19295740127563477, 0.28408804535865784, 0.14330795407295227, 0.37540408968925476], [0.44521209597587585, 0.46443694829940796, 0.3860609233379364, 0.44381704926490784], [0.0067197661846876144, 0.017497947439551353, 0.053679388016462326, 0.36563804745674133], [0.4035365879535675, 0.4869360625743866, 0.32791388034820557, 0.3581191897392273], [0.4469825029373169, 0.48872873187065125, 0.40463507175445557, 0.44334647059440613]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4ef242bb413685eee99de473b7498e8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_08e7f55a6191d372e6957d8e86a505f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_99a69859bf9f78e8ecc21fab494bb318(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_af883edbc88aa4a27a88712df55348a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4776863753795624, 0.16447702050209045, 0.46797195076942444, 0.31123462319374084]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5266408d73ccc0697ea498ac8bd36305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1840790a7b1403d2b9286416fabe337
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bb2d04e236c7cb10dde89bc77e1c3eb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afa241997c3bfb754dd9737513cfd22b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3769500176b4dd1989ddd054c69b8c7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7581e8216c5be5f67dcdd52a446b8a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_701a8211c4beca37a03cd4b8c9445889(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.28428125381469727, 0.23450800776481628, 0.39175689220428467, 0.08472386002540588], [0.27609318494796753, 0.37431052327156067, 0.19524559378623962, 0.3412277400493622], [0.3563365042209625, 0.42140257358551025, 0.19709186255931854, 0.46257638931274414], [0.23152372241020203, 0.1443825364112854, 0.10816586762666702, 0.14873941242694855], [0.1411408931016922, 0.217192143201828, 0.16752371191978455, 0.08864147961139679]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f35fbe92ff687f6a5a4ea4bb7f103da9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_aab4a00095558ca100c5f75bd249c500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1092fdae8695fda38d4206a34ccfaec7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_286de5f5fd1e4ebc2246b5e99a1ce2d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.13793407380580902, 0.4475548565387726, 0.046720605343580246, 0.4615541696548462], [0.1979195922613144, 0.4278280436992645, 0.2857111394405365, 0.4901394248008728], [0.0998683050274849, 0.44747817516326904, 0.19351927936077118, 0.3569736182689667], [0.21947908401489258, 0.33061710000038147, 0.14423230290412903, 0.3508850634098053], [0.41732439398765564, 0.47952800989151, 0.3923713266849518, 0.42253607511520386], [0.16849268972873688, 0.23413725197315216, 0.4728979170322418, 0.07528649270534515], [0.22205409407615662, 0.05290050059556961, 0.3209162652492523, 0.23186315596103668]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7f486c46a2fe7496c82058e91e357908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_34f9b6626a200ccee5dd10e879668edf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_da1787b0f94e4c84bbff124fe58ddee0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_dcbc95d62552f94e7b3a7fea2a669031(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.32870563864707947, 0.3947364389896393, 0.4261138439178467, 0.2962327301502228], [0.08479810506105423, 0.47901445627212524, 0.17515593767166138, 0.06920957565307617], [0.059684742242097855, 0.28597113490104675, 0.35019347071647644, 0.31610623002052307], [0.014615016989409924, 0.4546843469142914, 0.21722984313964844, 0.03398921713232994], [0.3845198452472687, 0.24070708453655243, 0.29894471168518066, 0.3289848864078522], [0.09035274386405945, 0.002472453750669956, 0.3865436017513275, 0.1406315714120865], [0.2091977745294571, 0.49779683351516724, 0.09025917947292328, 0.3644822835922241]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_56d2c6c50a6b8931450946a49d444a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_aa24a619441e29b70c993a866e6aec16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_be5935a4cf26121fcae7b5b53e3db9cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0a29b615738e017194de264793c5fe0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.49056458473205566, 0.4809737205505371, 0.35988709330558777, 0.12755186855793]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c9f3679cc991ff2ab4af6ccd1458ca41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1840790a7b1403d2b9286416fabe337
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a9ab4c7e9f311d93c5db916f18b585d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afa241997c3bfb754dd9737513cfd22b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_32fa6b4103245f9e8e6a85d86298494c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7581e8216c5be5f67dcdd52a446b8a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_29ffc1b820a7df0ef0246517d1c3250f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c13c07819324140bd151d939d6b61a18
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_76b3a76e882cd8e98256a021949d2aa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2db490ae667f38bd6d576336953f314b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fc583b51546d64afdfff7668fa2e9c57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09d4845a4d9dd963811e5ee784b5407a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_15787d7da144359d2d4aa70981f57ab3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1f2a5dbdb8d06b7586bdf9b8ea974b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_167681bcb3bc15543cad81ef6180fd02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.30826058983802795, 0.16620753705501556, 0.3658806383609772, 0.36707037687301636], [0.11164671927690506, 0.36271995306015015, 0.4650300145149231, 0.4107046127319336], [0.07298780232667923, 0.4408380389213562, 0.14397777616977692, 0.15549598634243011], [0.040000829845666885, 0.08228027075529099, 0.292710542678833, 0.036442629992961884], [0.4682316482067108, 0.4430815577507019, 0.376697838306427, 0.30927401781082153]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f35fbe92ff687f6a5a4ea4bb7f103da9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_aab4a00095558ca100c5f75bd249c500(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1092fdae8695fda38d4206a34ccfaec7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3956f1975a427c3afbdd4c120eb7a179(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3571544885635376, 0.42178237438201904, 0.09802666306495667, 0.4052501916885376]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ead5030757c66981087e7b8169910e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_021fb7726f8f5914e14552e79682ba97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_89f3bc5ea956b90d8d47af681d52e158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3a278e507712d3f7fc8c3c1c45861b61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4320468604564667, 0.3535347282886505, 0.12520934641361237, 0.23588906228542328]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4caf997155301f0c25e86da14f20db43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1840790a7b1403d2b9286416fabe337
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6ddd687fde1b06882b206354d1087de6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afa241997c3bfb754dd9737513cfd22b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4e1ba5248cd0ccc90922545c42c33716(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7581e8216c5be5f67dcdd52a446b8a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_116b004b267395b3ad1f825045944043(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c9f3679cc991ff2ab4af6ccd1458ca41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1840790a7b1403d2b9286416fabe337
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a9ab4c7e9f311d93c5db916f18b585d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afa241997c3bfb754dd9737513cfd22b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_32fa6b4103245f9e8e6a85d86298494c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7581e8216c5be5f67dcdd52a446b8a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_eeba2a01ac69d8124e958e02e9b91ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c13c07819324140bd151d939d6b61a18
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a417d93dc6f2a12cac79354d2407c583(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2db490ae667f38bd6d576336953f314b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_07bfb4b440fdc1b489ddc89bd5a1575a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09d4845a4d9dd963811e5ee784b5407a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_50eadf98b7f32ef6b46124604f3f0e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1f2a5dbdb8d06b7586bdf9b8ea974b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fd4d1acee39d40d9fa825d798fba93ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.37920650839805603, 0.32739609479904175, 0.01771315187215805, 0.4222439229488373]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b805aeefb3ff9009efb5bc1c84e0a78b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1840790a7b1403d2b9286416fabe337
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_38358845f33b51b6130e3b8448f90512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afa241997c3bfb754dd9737513cfd22b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ca95aedb0edacb8602ed8a780d120610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7581e8216c5be5f67dcdd52a446b8a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f181c082e4aabfb6ff934f4270f53430(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.120260089635849, 0.10687462985515594, 0.4020973742008209, 0.1944931298494339], [0.14424557983875275, 0.34658628702163696, 0.2673088014125824, 0.49299418926239014], [0.4701901376247406, 0.2641819417476654, 0.31320127844810486, 0.33311450481414795], [0.037186916917562485, 0.23197132349014282, 0.4214165210723877, 0.09179126471281052], [0.10999499261379242, 0.04785877466201782, 0.09373696148395538, 0.3439233601093292], [0.3809078335762024, 0.11442305147647858, 0.10039916634559631, 0.17913693189620972], [0.41799628734588623, 0.3991064727306366, 0.35630565881729126, 0.16492298245429993]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7f486c46a2fe7496c82058e91e357908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a4ac308c465319518eb8c55c193cda2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_34f9b6626a200ccee5dd10e879668edf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d4394fade0b6bd532d6f949492cf6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_da1787b0f94e4c84bbff124fe58ddee0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cdc757c55a6e58fa8c469d5ff75e982
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a31e0009fb4e854faa951271ad512fe5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11187709122896194, 0.40898066759109497, 0.25907111167907715, 0.42682981491088867], [0.1378955841064453, 0.0975421741604805, 0.119081050157547, 0.19117246568202972], [0.07604412734508514, 0.02066691406071186, 0.4777871072292328, 0.4768635928630829], [0.4223090708255768, 0.32861530780792236, 0.30099281668663025, 0.1979154497385025], [0.48413437604904175, 0.29007235169410706, 0.49729445576667786, 0.22377794981002808], [0.4088941812515259, 0.1840423196554184, 0.017735710367560387, 0.06378057599067688]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_66fc1a02415f1d43ed44a9fc9bfedc94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1840790a7b1403d2b9286416fabe337
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_24b320a8b568be9bbfa55e68974931a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afa241997c3bfb754dd9737513cfd22b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bac53e74653b2022a2f7ed625cb90cd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7581e8216c5be5f67dcdd52a446b8a6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_7c081652e4b17e46c909fcac7ff1eb86(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 144, 216], dtype='float32'),
                paddle.static.InputSpec(shape=[300, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0fab934e84106cca2a255add5d26bf0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c081652e4b17e46c909fcac7ff1eb86
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_7a44693291b4f75f641260caf74a1595(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 72, 108], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3dcea6ef50dad75d95a86211dff9814(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a44693291b4f75f641260caf74a1595
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_86503c6d5bfcc7fc36a245e005ebf5ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 36, 54], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_089a83198c9a9f57fdd33a379d759d15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86503c6d5bfcc7fc36a245e005ebf5ac
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_01819e71d6f28abcf3b1fca8de0c41bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 18, 27], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0503a4439a1f503b74c2f92585a318e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01819e71d6f28abcf3b1fca8de0c41bc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_8765ff9ccecd585b39401af99d7abb70(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
                paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_952bdef9e9888c2b91f370d53f5127ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8765ff9ccecd585b39401af99d7abb70
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d83b891c953842b06480244ada63d520(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 88, 132], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5a06088c138f893c10c1e02187258f46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d83b891c953842b06480244ada63d520
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_5f1606f57b9b276034b0c2c20ef8d36c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 44, 66], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6510a33d986939139078e18dd4fa9129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f1606f57b9b276034b0c2c20ef8d36c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_3a4102f1d4760d33d167a2692f7f1a37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 22, 33], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a2434d25ec8303c4fff9dcdd82d7916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a4102f1d4760d33d167a2692f7f1a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_41774258b4b7d8f8cfd0123ed306e87d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 168, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_296e837befdeac909dce4528a83f929b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41774258b4b7d8f8cfd0123ed306e87d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.173097625374794, 0.24690791964530945, 0.12752406299114227, 0.21726328134536743], [0.42141810059547424, 0.32032057642936707, 0.22069108486175537, 0.25408151745796204]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_da202f64daa6dc05a9c87d82a05108c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 84, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34b2743993285ffa56604d6d868926b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da202f64daa6dc05a9c87d82a05108c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_bc10bec778ede066a5e22ab9b386c6e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 42, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c99e2199e10237e00917f9ff312e87f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc10bec778ede066a5e22ab9b386c6e1
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_533dffd29bd1bc4cee8e1ae0763c4025(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 21, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3a8d2e8794afcb4558d16a74d54f997(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_533dffd29bd1bc4cee8e1ae0763c4025
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_58e7a6930d535dd552a383176ead2e15(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
                paddle.static.InputSpec(shape=[100, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ce6cda6cd90ab223df6cfbab8a88c56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58e7a6930d535dd552a383176ead2e15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_73f019197a7ede3b3404f5b720ab62ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 88, 132], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1eacc7747d72499be36721a478b539b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73f019197a7ede3b3404f5b720ab62ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d423ecc10e49d53abf1b910943560e99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 44, 66], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62478dbc3eb3da2e41a0dfaec6463bf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d423ecc10e49d53abf1b910943560e99
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_c0522ea11ef9a4f02022eb2a9befcfcb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 22, 33], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f79e4b749b17bf546a9a4c7924985832(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0522ea11ef9a4f02022eb2a9befcfcb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_8f4ea02cf8c5a0bdbe7cfc47596c71e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 136, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1023f91f9ba432ba977ea8210686159d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f4ea02cf8c5a0bdbe7cfc47596c71e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4688277244567871, 0.30018413066864014, 0.23425810039043427, 0.3811199963092804], [0.4602409303188324, 0.1876598298549652, 0.14904391765594482, 0.10620903223752975]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_a12575008a0c309d2f6ce3d51ae1ba49(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 68, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_daf38d38123fa665351d3650e3bb4e69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a12575008a0c309d2f6ce3d51ae1ba49
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_8a8ff3b6c31157a113474802d7e6c27c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 34, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c7ea9e1ed8f1823c6b0c83d0faaf32aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a8ff3b6c31157a113474802d7e6c27c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_3c6029b1dc71f391ea48bdb65ded9d84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 17, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_222cae5261505643cc5740d94166d504(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c6029b1dc71f391ea48bdb65ded9d84
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_0e72fbafecc7a71a26b51ac41608c561(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 200, 304], dtype='float32'),
                paddle.static.InputSpec(shape=[2, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37dc26e1709b88534fab3ec1680a3b45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e72fbafecc7a71a26b51ac41608c561
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11911441385746002, 0.12901979684829712, 0.18489500880241394, 0.019981015473604202], [0.04435403645038605, 0.22364260256290436, 0.20171409845352173, 0.3678695559501648]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_bbd4ba2b4c09815d84517611259a729e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9a7925101162ce70929bbb2d439eb53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbd4ba2b4c09815d84517611259a729e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_12d33a52199c82c1c2d2da45c918768c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e239aea1c5961f664149b5389e7f114f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d33a52199c82c1c2d2da45c918768c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_1cd978613ed83f1399af2358de9fccfa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2024f98a19ccb76826f49c07202cea74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cd978613ed83f1399af2358de9fccfa
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_952bdef9e9888c2b91f370d53f5127ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8765ff9ccecd585b39401af99d7abb70
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5a06088c138f893c10c1e02187258f46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d83b891c953842b06480244ada63d520
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6510a33d986939139078e18dd4fa9129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f1606f57b9b276034b0c2c20ef8d36c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9a2434d25ec8303c4fff9dcdd82d7916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a4102f1d4760d33d167a2692f7f1a37
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_081d0e74c33d435e60ffc3c2cb720a51(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 192, 288], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa052ce9f05b77803c2681a2f22e74d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_081d0e74c33d435e60ffc3c2cb720a51
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.23170973360538483, 0.44410353899002075, 0.197391077876091, 0.09819187223911285], [0.21788160502910614, 0.20119178295135498, 0.13865119218826294, 0.3262018859386444], [0.0408363938331604, 0.09758158028125763, 0.4664136469364166, 0.2251659631729126], [0.04279877617955208, 0.16735155880451202, 0.17869096994400024, 0.45279648900032043], [0.016807518899440765, 0.10631626099348068, 0.23214693367481232, 0.3479967713356018], [0.2931079566478729, 0.11825201660394669, 0.17887479066848755, 0.19634494185447693], [0.3446366786956787, 0.0214198250323534, 0.029282689094543457, 0.4609162211418152]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_fe5f00c7795f813938646d6dfa46cbde(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cff2e1ec9c9fc398370c52b8c05692b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe5f00c7795f813938646d6dfa46cbde
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_de7b8e0091308fb2623d94801898510a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b851880d4f53edd34e3e6c5ef7fec91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de7b8e0091308fb2623d94801898510a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f9344e4cc4a38632a81efc0324842919(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 24, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a23152fa82301c60a0220db3fffa8ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9344e4cc4a38632a81efc0324842919
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_4c842a517c20adfc2f894842689fe4df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63c150df5a8dc3a69eb7c291e8aa3d23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c842a517c20adfc2f894842689fe4df
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.17230640351772308, 0.4015914499759674, 0.22270041704177856, 0.4950568675994873], [0.014866560697555542, 0.2018941342830658, 0.02957601472735405, 0.3901946544647217], [0.46414849162101746, 0.42340996861457825, 0.007422108668833971, 0.24007540941238403], [0.3502897024154663, 0.4091551899909973, 0.3867229223251343, 0.4222806394100189], [0.2405211180448532, 0.34306421875953674, 0.013049566186964512, 0.28451433777809143], [0.4376644194126129, 0.09262708574533463, 0.41278737783432007, 0.17460638284683228]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_980b15fc98a7dc0869fd184c37560c8d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 80, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68dee6e30f4581c263d4843a032104af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_980b15fc98a7dc0869fd184c37560c8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_be3b22d1271317d57ffa8b5874542953(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 40, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_570a64c8e53f61d7454701fb90f6e115(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be3b22d1271317d57ffa8b5874542953
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_5756be2528990e1b93cd152a3f1ee30e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 20, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51aa8196fc819021f348015c7b28b67e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5756be2528990e1b93cd152a3f1ee30e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_0d20704158529c8e76be8812ebdaf66e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 200, 272], dtype='float32'),
                paddle.static.InputSpec(shape=[3, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8416606249208fa5e5b04a6ece34eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d20704158529c8e76be8812ebdaf66e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.168807253241539, 0.24565082788467407, 0.4337322413921356, 0.09575147181749344], [0.4107934534549713, 0.29058510065078735, 0.46093037724494934, 0.3507699966430664], [0.0019927481189370155, 0.0774812400341034, 0.02200215682387352, 0.1899755746126175]], dtype='float32').reshape([3, 4]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_6240be8123b842ca06e5d9e7582597b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 100, 136], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_097dcabda76bfdb1c3cee193bcc8876c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6240be8123b842ca06e5d9e7582597b1
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ff08befe72043e5edc875ea2ada35d44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 50, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bba3d875a1d21175328de21468564c4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ff08befe72043e5edc875ea2ada35d44
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_9f23df2573c628c2f1efb05536a2e6cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 25, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72266937f263b6cad1bf399c908435e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f23df2573c628c2f1efb05536a2e6cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_af5f31ac20d293797aea976638a6aa21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e72fbafecc7a71a26b51ac41608c561
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.12799687683582306, 0.006959439255297184, 0.43831005692481995, 0.14842861890792847], [0.18375302851200104, 0.45857229828834534, 0.2913435101509094, 0.38440775871276855]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d9a7925101162ce70929bbb2d439eb53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbd4ba2b4c09815d84517611259a729e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e239aea1c5961f664149b5389e7f114f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12d33a52199c82c1c2d2da45c918768c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2024f98a19ccb76826f49c07202cea74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cd978613ed83f1399af2358de9fccfa
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f0eb783dabfcd8a21f0a79de757eae1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 168, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd3d56816196c3d60207779671a9e9da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0eb783dabfcd8a21f0a79de757eae1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.011920098215341568, 0.4075399935245514, 0.2567863166332245, 0.11302003264427185]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_dc2871c89d93bf63809c66a5e89920ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 84, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b3b7939241e5598b29504e123e1460a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc2871c89d93bf63809c66a5e89920ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_3ce717403b70fa3e77ebf70859d06432(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 42, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fef17690a6f1916c4eb682311116f23d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ce717403b70fa3e77ebf70859d06432
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e1fceb1de7bfd8bbc9b61d39cf7715fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 21, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ed2d26619e4c472ae2abcdb175e630f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1fceb1de7bfd8bbc9b61d39cf7715fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_990d6e8bf7faa38cc7deeb9ba665afeb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 136, 208], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74d08cdf7182a57ff9aaec488522df39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_990d6e8bf7faa38cc7deeb9ba665afeb
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.03101678565144539, 0.274365097284317, 0.00527990935370326, 0.19069573283195496], [0.058812227100133896, 0.1637631058692932, 0.025367382913827896, 0.07655315101146698], [0.19295740127563477, 0.28408804535865784, 0.14330795407295227, 0.37540408968925476], [0.44521209597587585, 0.46443694829940796, 0.3860609233379364, 0.44381704926490784], [0.0067197661846876144, 0.017497947439551353, 0.053679388016462326, 0.36563804745674133], [0.4035365879535675, 0.4869360625743866, 0.32791388034820557, 0.3581191897392273], [0.4469825029373169, 0.48872873187065125, 0.40463507175445557, 0.44334647059440613]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_284641b414dcea109caa99ab86826114(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 68, 104], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_481e282d45ee9f9d080288a3bc6aca2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_284641b414dcea109caa99ab86826114
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_1a7bb4b3693387b30d54d0cbeb8fddd6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 34, 52], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ed56756918b08047e1d8ae28dfc337d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a7bb4b3693387b30d54d0cbeb8fddd6
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ad2691a3ef8a2e2fed4df2ecfe1fa7ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 64, 17, 26], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2e833e5a21511c5f8758351f9c3f2603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad2691a3ef8a2e2fed4df2ecfe1fa7ef
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_92442a46b544acfbbf212198e940afe9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0cda5c5472820d6905f26893c4425d9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92442a46b544acfbbf212198e940afe9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4776863753795624, 0.16447702050209045, 0.46797195076942444, 0.31123462319374084]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_68dee6e30f4581c263d4843a032104af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_980b15fc98a7dc0869fd184c37560c8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_570a64c8e53f61d7454701fb90f6e115(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be3b22d1271317d57ffa8b5874542953
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_51aa8196fc819021f348015c7b28b67e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5756be2528990e1b93cd152a3f1ee30e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_a9edd85f3c2f8cb1be509a96c605c005(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 184, 280], dtype='float32'),
                paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_517cefbe6cbddd7db7d06bb283eead10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9edd85f3c2f8cb1be509a96c605c005
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.28428125381469727, 0.23450800776481628, 0.39175689220428467, 0.08472386002540588], [0.27609318494796753, 0.37431052327156067, 0.19524559378623962, 0.3412277400493622], [0.3563365042209625, 0.42140257358551025, 0.19709186255931854, 0.46257638931274414], [0.23152372241020203, 0.1443825364112854, 0.10816586762666702, 0.14873941242694855], [0.1411408931016922, 0.217192143201828, 0.16752371191978455, 0.08864147961139679]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_45a8ebe693a2f540f30e9d40f00dffc2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 92, 140], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ae4e8bf89a18694500fbe7866aac383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45a8ebe693a2f540f30e9d40f00dffc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_0ab530697238f3b4a08fd9d7002976e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 46, 70], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49bc865cf4ec7701f064d83b6b517f4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ab530697238f3b4a08fd9d7002976e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_202fdf0deb4e5b24afa22314f3aab34a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 23, 35], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0606689ea20e3028c3e46082a616d446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202fdf0deb4e5b24afa22314f3aab34a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b1bde784650e08128960e8eb453cff7c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 160, 240], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eef304bd1889848cc515082f34adfb9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1bde784650e08128960e8eb453cff7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.13793407380580902, 0.4475548565387726, 0.046720605343580246, 0.4615541696548462], [0.1979195922613144, 0.4278280436992645, 0.2857111394405365, 0.4901394248008728], [0.0998683050274849, 0.44747817516326904, 0.19351927936077118, 0.3569736182689667], [0.21947908401489258, 0.33061710000038147, 0.14423230290412903, 0.3508850634098053], [0.41732439398765564, 0.47952800989151, 0.3923713266849518, 0.42253607511520386], [0.16849268972873688, 0.23413725197315216, 0.4728979170322418, 0.07528649270534515], [0.22205409407615662, 0.05290050059556961, 0.3209162652492523, 0.23186315596103668]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_494ef62f9faa6c36ad85ad0bb9cd0e8b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 80, 120], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1661c4fa80fac79e57b586aff6c25204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_494ef62f9faa6c36ad85ad0bb9cd0e8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_6effcf6d15958452c78b365f0b5122bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 40, 60], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0b8f658d2e9924bc25df127c37685ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6effcf6d15958452c78b365f0b5122bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_c5143bcb351bde571eca647f193a66c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 20, 30], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a55d0ba0c8891dade29db3b85879c01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5143bcb351bde571eca647f193a66c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8664b5cf2fed0394ea8d86077066ae03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_081d0e74c33d435e60ffc3c2cb720a51
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.32870563864707947, 0.3947364389896393, 0.4261138439178467, 0.2962327301502228], [0.08479810506105423, 0.47901445627212524, 0.17515593767166138, 0.06920957565307617], [0.059684742242097855, 0.28597113490104675, 0.35019347071647644, 0.31610623002052307], [0.014615016989409924, 0.4546843469142914, 0.21722984313964844, 0.03398921713232994], [0.3845198452472687, 0.24070708453655243, 0.29894471168518066, 0.3289848864078522], [0.09035274386405945, 0.002472453750669956, 0.3865436017513275, 0.1406315714120865], [0.2091977745294571, 0.49779683351516724, 0.09025917947292328, 0.3644822835922241]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cff2e1ec9c9fc398370c52b8c05692b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe5f00c7795f813938646d6dfa46cbde
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5b851880d4f53edd34e3e6c5ef7fec91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_de7b8e0091308fb2623d94801898510a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a23152fa82301c60a0220db3fffa8ab4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9344e4cc4a38632a81efc0324842919
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f0d7ac0b2cb53a2975d08caaad115f5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d46128dbd9dfc98a846e344c83365dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d7ac0b2cb53a2975d08caaad115f5f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.49056458473205566, 0.4809737205505371, 0.35988709330558777, 0.12755186855793]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ac55e82f2521596624db1deabb541c01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 88, 132], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3d594419e4a449b53ef109d489328d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac55e82f2521596624db1deabb541c01
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_a2ac5e97ad9d172cae6a32a279816153(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 44, 66], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7965a2e3cf0f7a9d12b5b39c6d610704(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2ac5e97ad9d172cae6a32a279816153
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f8ac9a1ba1ebf8ee8a4c317b78c48b94(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 22, 33], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bceef8bfae7b8d3c85ed9a8c98a95c75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8ac9a1ba1ebf8ee8a4c317b78c48b94
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0fab934e84106cca2a255add5d26bf0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c081652e4b17e46c909fcac7ff1eb86
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d3dcea6ef50dad75d95a86211dff9814(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a44693291b4f75f641260caf74a1595
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_089a83198c9a9f57fdd33a379d759d15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86503c6d5bfcc7fc36a245e005ebf5ac
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0503a4439a1f503b74c2f92585a318e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01819e71d6f28abcf3b1fca8de0c41bc
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a4bbec74d37a575dc0e6cfb60cd6f5b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9edd85f3c2f8cb1be509a96c605c005
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.30826058983802795, 0.16620753705501556, 0.3658806383609772, 0.36707037687301636], [0.11164671927690506, 0.36271995306015015, 0.4650300145149231, 0.4107046127319336], [0.07298780232667923, 0.4408380389213562, 0.14397777616977692, 0.15549598634243011], [0.040000829845666885, 0.08228027075529099, 0.292710542678833, 0.036442629992961884], [0.4682316482067108, 0.4430815577507019, 0.376697838306427, 0.30927401781082153]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5ae4e8bf89a18694500fbe7866aac383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45a8ebe693a2f540f30e9d40f00dffc2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_49bc865cf4ec7701f064d83b6b517f4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ab530697238f3b4a08fd9d7002976e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0606689ea20e3028c3e46082a616d446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_202fdf0deb4e5b24afa22314f3aab34a
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_52909d40df3d7c374d43d932c6d2d06c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 176, 176], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ef6830fd0f95021d054ea600f2631cd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52909d40df3d7c374d43d932c6d2d06c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3571544885635376, 0.42178237438201904, 0.09802666306495667, 0.4052501916885376]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_27eec0e5654f3e93b6bc2e1b22809e94(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 88, 88], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bef618db351d21f105502166d173d48d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27eec0e5654f3e93b6bc2e1b22809e94
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d6530bf2550a1a8f2592d216bc04fcdd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 44, 44], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c9e3c61b43a6dbca27cb5c467f95fd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6530bf2550a1a8f2592d216bc04fcdd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_64b3748386cab310c5411152965fb105(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 22, 22], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_db4e188e1baff6ba43d84244369f263c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_64b3748386cab310c5411152965fb105
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b700f528f4cd736d657f23fac20d8ab2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 200, 304], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a54bd77a1fa5ef7c7d0484bc5838019(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b700f528f4cd736d657f23fac20d8ab2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4320468604564667, 0.3535347282886505, 0.12520934641361237, 0.23588906228542328]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_a0a069a72ee1f325e8863e89de8cd1a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7eedc7251f18e7105adb11c32221ae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0a069a72ee1f325e8863e89de8cd1a3
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_b864cf84b0998f8b005386648f2e3b81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1293359c450604889fbd342fd168b70c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b864cf84b0998f8b005386648f2e3b81
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_a78cfb25c03dfe525ed8ec4a2524436c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_843428d31b1d090ee3211d05d081e8c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a78cfb25c03dfe525ed8ec4a2524436c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_01191e6cab4948c92897ad0fbd773546(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 176, 264], dtype='float32'),
                paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e81b4c7efa24a8866a54f947937f4fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01191e6cab4948c92897ad0fbd773546
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b3d594419e4a449b53ef109d489328d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ac55e82f2521596624db1deabb541c01
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7965a2e3cf0f7a9d12b5b39c6d610704(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2ac5e97ad9d172cae6a32a279816153
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bceef8bfae7b8d3c85ed9a8c98a95c75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f8ac9a1ba1ebf8ee8a4c317b78c48b94
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0ce6cda6cd90ab223df6cfbab8a88c56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_58e7a6930d535dd552a383176ead2e15
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1eacc7747d72499be36721a478b539b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73f019197a7ede3b3404f5b720ab62ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_62478dbc3eb3da2e41a0dfaec6463bf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d423ecc10e49d53abf1b910943560e99
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f79e4b749b17bf546a9a4c7924985832(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0522ea11ef9a4f02022eb2a9befcfcb
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_e6e63fc367a42a84ff3be03bdcde3162(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 192, 288], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6aee3e3f8382b0103c82a484307a874(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e63fc367a42a84ff3be03bdcde3162
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.37920650839805603, 0.32739609479904175, 0.01771315187215805, 0.4222439229488373]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_308242c3685d5e05281ce87674d8dfaf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 96, 144], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0eed8285eb6f974e766a73948ab21121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_308242c3685d5e05281ce87674d8dfaf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_abc086b63d44fbf66b6beb2588596f70(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 48, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e98e2197169ccd388517738ed6d80d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abc086b63d44fbf66b6beb2588596f70
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_77d9275544bc3887bb4f5c911f372ba9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 24, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[0, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3fd170430df27925362e7e35e78bf01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77d9275544bc3887bb4f5c911f372ba9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_814acfb06bb61ace260685846ee9a1eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1bde784650e08128960e8eb453cff7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.120260089635849, 0.10687462985515594, 0.4020973742008209, 0.1944931298494339], [0.14424557983875275, 0.34658628702163696, 0.2673088014125824, 0.49299418926239014], [0.4701901376247406, 0.2641819417476654, 0.31320127844810486, 0.33311450481414795], [0.037186916917562485, 0.23197132349014282, 0.4214165210723877, 0.09179126471281052], [0.10999499261379242, 0.04785877466201782, 0.09373696148395538, 0.3439233601093292], [0.3809078335762024, 0.11442305147647858, 0.10039916634559631, 0.17913693189620972], [0.41799628734588623, 0.3991064727306366, 0.35630565881729126, 0.16492298245429993]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1661c4fa80fac79e57b586aff6c25204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_494ef62f9faa6c36ad85ad0bb9cd0e8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f0b8f658d2e9924bc25df127c37685ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6effcf6d15958452c78b365f0b5122bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a55d0ba0c8891dade29db3b85879c01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5143bcb351bde571eca647f193a66c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_6161b6048ce7d7a53519fa392e78cc17(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 168, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81b437fec7d1563296f3f72b54e5b100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6161b6048ce7d7a53519fa392e78cc17
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11187709122896194, 0.40898066759109497, 0.25907111167907715, 0.42682981491088867], [0.1378955841064453, 0.0975421741604805, 0.119081050157547, 0.19117246568202972], [0.07604412734508514, 0.02066691406071186, 0.4777871072292328, 0.4768635928630829], [0.4223090708255768, 0.32861530780792236, 0.30099281668663025, 0.1979154497385025], [0.48413437604904175, 0.29007235169410706, 0.49729445576667786, 0.22377794981002808], [0.4088941812515259, 0.1840423196554184, 0.017735710367560387, 0.06378057599067688]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1b3b7939241e5598b29504e123e1460a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc2871c89d93bf63809c66a5e89920ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fef17690a6f1916c4eb682311116f23d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ce717403b70fa3e77ebf70859d06432
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7ed2d26619e4c472ae2abcdb175e630f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1fceb1de7bfd8bbc9b61d39cf7715fa
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_fedf488a264409407e548188bad1d830(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2336670ef5a79082b952f115ea75a37e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fedf488a264409407e548188bad1d830
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_a383e06e16f6eebfe8028e6b10796710(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dcc04a19ead5aac0ed2af4fbd418155c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a383e06e16f6eebfe8028e6b10796710
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_4a649f60367e99c964bbd60f4e35d477(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ac15813e5431c988929a3e65846993b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a649f60367e99c964bbd60f4e35d477
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ab828bb8cf502c343e7fcc65e62a9c98(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 2, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2cf88bcb6beb09cf59b28865734968d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab828bb8cf502c343e7fcc65e62a9c98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_3b98ba588443913377f02b5d8facdaea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdd78221f9cdb83b2be5a4f5ed1647ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_fd66776ca8bd45922283de8725111a52(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba231bb920d388e1c446d6ac5b243113(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_15438c4973df542e48bc7e701de9af02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_167e1fb781515b0948bfa40068e079ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f7fe477b49a132218570839021cf8e57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 7, 7, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_563ab4533371d1ec59e48afcb4e3aceb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c92c64b2c7255a8f93646428f55fdf61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.173097625374794, 0.24690791964530945, 0.12752406299114227, 0.21726328134536743], [0.42141810059547424, 0.32032057642936707, 0.22069108486175537, 0.25408151745796204]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2acd64ac8ccaf751f9d446560747286e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c014fc6dd07ccf074032de349550ee5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d27afb3516dd8eaee8b482ffbca62d66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ce1001b41e8feeed4b7ae1ef038bf991(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fedf488a264409407e548188bad1d830
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4b03e5b65348feb9a43c5df4748d6cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a383e06e16f6eebfe8028e6b10796710
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_167627ed408a304f576f27e5ddc80a1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a649f60367e99c964bbd60f4e35d477
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_362bc4eedfaa62ab44852f437aca7eac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab828bb8cf502c343e7fcc65e62a9c98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1b09dd395930cf2d9b3881f9f70c652c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4688277244567871, 0.30018413066864014, 0.23425810039043427, 0.3811199963092804], [0.4602409303188324, 0.1876598298549652, 0.14904391765594482, 0.10620903223752975]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fc2835ba9704a06c98ca204011f3fdfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_920ab84fb791557ad5769a002e10c8a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5ce2634f07312a6b7b09cb1242cbbeb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0cd50d9c853aa58e257f5d8540fa0a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11911441385746002, 0.12901979684829712, 0.18489500880241394, 0.019981015473604202], [0.04435403645038605, 0.22364260256290436, 0.20171409845352173, 0.3678695559501648]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c24f45157ebf9d965a852f266ac080e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6f40b06c6fd81a50bcf39532419ea713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_eb71953d0b8c9306738a88560ab538d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cdd78221f9cdb83b2be5a4f5ed1647ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ba231bb920d388e1c446d6ac5b243113(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_167e1fb781515b0948bfa40068e079ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_563ab4533371d1ec59e48afcb4e3aceb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a5509d0bd050807542a15b98bd48d381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.23170973360538483, 0.44410353899002075, 0.197391077876091, 0.09819187223911285], [0.21788160502910614, 0.20119178295135498, 0.13865119218826294, 0.3262018859386444], [0.0408363938331604, 0.09758158028125763, 0.4664136469364166, 0.2251659631729126], [0.04279877617955208, 0.16735155880451202, 0.17869096994400024, 0.45279648900032043], [0.016807518899440765, 0.10631626099348068, 0.23214693367481232, 0.3479967713356018], [0.2931079566478729, 0.11825201660394669, 0.17887479066848755, 0.19634494185447693], [0.3446366786956787, 0.0214198250323534, 0.029282689094543457, 0.4609162211418152]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fc638627cb1807bf15cd56360cace157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_17cd40d19a0ad8dcb7b16948b4ff336e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3d4be74828ad11a885339423038ffef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.25, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72ee291827f662b32d7162300f9a8f3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.17230640351772308, 0.4015914499759674, 0.22270041704177856, 0.4950568675994873], [0.014866560697555542, 0.2018941342830658, 0.02957601472735405, 0.3901946544647217], [0.46414849162101746, 0.42340996861457825, 0.007422108668833971, 0.24007540941238403], [0.3502897024154663, 0.4091551899909973, 0.3867229223251343, 0.4222806394100189], [0.2405211180448532, 0.34306421875953674, 0.013049566186964512, 0.28451433777809143], [0.4376644194126129, 0.09262708574533463, 0.41278737783432007, 0.17460638284683228]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_ec35c25bea37ebcc9d1750caf8882e62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99a8cf9973107c51f5313f04fce9c8f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec35c25bea37ebcc9d1750caf8882e62
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_a84e4b50450ac284ca75c853ae8663be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.0625, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5bd82d44359460b034dcac158ddd6d40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a84e4b50450ac284ca75c853ae8663be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f89971896f179f169fbd26c9a499384c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.roi_align(input_0, input_1, input_2, 14, 14, 0.03125, 0, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04526012a781cdfc5f3544929a251205(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f89971896f179f169fbd26c9a499384c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8ed8f79afe0f2faff9b49a71f77b7987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.168807253241539, 0.24565082788467407, 0.4337322413921356, 0.09575147181749344], [0.4107934534549713, 0.29058510065078735, 0.46093037724494934, 0.3507699966430664], [0.0019927481189370155, 0.0774812400341034, 0.02200215682387352, 0.1899755746126175]], dtype='float32').reshape([3, 4]),
                paddle.to_tensor([3], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7cf13386cc34d943a6104896046aac45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_936a5b189973efa1d908fb3426d451df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9c785b2679638f1e9049448e6fe8ce57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_21ae8f302de62e57b2241f75ed80a4c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.12799687683582306, 0.006959439255297184, 0.43831005692481995, 0.14842861890792847], [0.18375302851200104, 0.45857229828834534, 0.2913435101509094, 0.38440775871276855]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c24f45157ebf9d965a852f266ac080e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6f40b06c6fd81a50bcf39532419ea713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_eb71953d0b8c9306738a88560ab538d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ef1f9c524bc9fde46623c5f285fa8ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.011920098215341568, 0.4075399935245514, 0.2567863166332245, 0.11302003264427185]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d59ad5d1f19e30f11e441d5bfbd31bdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec35c25bea37ebcc9d1750caf8882e62
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1081fbeca053a156c5ed60c656824a51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a84e4b50450ac284ca75c853ae8663be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6c66cc5a689fe16eaf5c2c89e8b719fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f89971896f179f169fbd26c9a499384c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6145a73a4ec5ff446beb6dcbf936777f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.03101678565144539, 0.274365097284317, 0.00527990935370326, 0.19069573283195496], [0.058812227100133896, 0.1637631058692932, 0.025367382913827896, 0.07655315101146698], [0.19295740127563477, 0.28408804535865784, 0.14330795407295227, 0.37540408968925476], [0.44521209597587585, 0.46443694829940796, 0.3860609233379364, 0.44381704926490784], [0.0067197661846876144, 0.017497947439551353, 0.053679388016462326, 0.36563804745674133], [0.4035365879535675, 0.4869360625743866, 0.32791388034820557, 0.3581191897392273], [0.4469825029373169, 0.48872873187065125, 0.40463507175445557, 0.44334647059440613]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d2c073533c9d9ad0af71ef7ff0142e94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3cf001a25530a735a55b056d2cc41f1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b2b7267e5bbb42cfce080e19faf9b32e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4755766d243eb7a07d6fa48e64a1e317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4776863753795624, 0.16447702050209045, 0.46797195076942444, 0.31123462319374084]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_99a8cf9973107c51f5313f04fce9c8f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec35c25bea37ebcc9d1750caf8882e62
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5bd82d44359460b034dcac158ddd6d40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a84e4b50450ac284ca75c853ae8663be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_04526012a781cdfc5f3544929a251205(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f89971896f179f169fbd26c9a499384c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_bd583cf191f431a2980f4f6f340507cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.28428125381469727, 0.23450800776481628, 0.39175689220428467, 0.08472386002540588], [0.27609318494796753, 0.37431052327156067, 0.19524559378623962, 0.3412277400493622], [0.3563365042209625, 0.42140257358551025, 0.19709186255931854, 0.46257638931274414], [0.23152372241020203, 0.1443825364112854, 0.10816586762666702, 0.14873941242694855], [0.1411408931016922, 0.217192143201828, 0.16752371191978455, 0.08864147961139679]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f80f151f8cebe35b47c49d940efe0138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8fd2b4b2c2b093d4f7d543ddb32e9d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_404866aa237caae19890499faba0d12a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_64ebf591712f09b232ccb09128f77a14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.13793407380580902, 0.4475548565387726, 0.046720605343580246, 0.4615541696548462], [0.1979195922613144, 0.4278280436992645, 0.2857111394405365, 0.4901394248008728], [0.0998683050274849, 0.44747817516326904, 0.19351927936077118, 0.3569736182689667], [0.21947908401489258, 0.33061710000038147, 0.14423230290412903, 0.3508850634098053], [0.41732439398765564, 0.47952800989151, 0.3923713266849518, 0.42253607511520386], [0.16849268972873688, 0.23413725197315216, 0.4728979170322418, 0.07528649270534515], [0.22205409407615662, 0.05290050059556961, 0.3209162652492523, 0.23186315596103668]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_982d33a60e3bc08998502884855ce0f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2b820161a88956f2776a61a4b13394ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_32493094d0ec012c8f1a779c54fb08a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_10119bd5be59f9a545fdf36d4fe4e1c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.32870563864707947, 0.3947364389896393, 0.4261138439178467, 0.2962327301502228], [0.08479810506105423, 0.47901445627212524, 0.17515593767166138, 0.06920957565307617], [0.059684742242097855, 0.28597113490104675, 0.35019347071647644, 0.31610623002052307], [0.014615016989409924, 0.4546843469142914, 0.21722984313964844, 0.03398921713232994], [0.3845198452472687, 0.24070708453655243, 0.29894471168518066, 0.3289848864078522], [0.09035274386405945, 0.002472453750669956, 0.3865436017513275, 0.1406315714120865], [0.2091977745294571, 0.49779683351516724, 0.09025917947292328, 0.3644822835922241]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fc638627cb1807bf15cd56360cace157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_17cd40d19a0ad8dcb7b16948b4ff336e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3d4be74828ad11a885339423038ffef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5c15b8074dbf077a236315484c8bd47c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.49056458473205566, 0.4809737205505371, 0.35988709330558777, 0.12755186855793]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6253e3a80f16035f6da6589e6b1050e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec35c25bea37ebcc9d1750caf8882e62
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2c6c9620eea857f802de4ac30f79d74e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a84e4b50450ac284ca75c853ae8663be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1dd354175e3fba474e0d439d27fdc240(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f89971896f179f169fbd26c9a499384c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2336670ef5a79082b952f115ea75a37e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fedf488a264409407e548188bad1d830
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 144, 216], dtype='float32', min=0, max=0.5),
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([300], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_dcc04a19ead5aac0ed2af4fbd418155c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a383e06e16f6eebfe8028e6b10796710
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0ac15813e5431c988929a3e65846993b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a649f60367e99c964bbd60f4e35d477
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b2cf88bcb6beb09cf59b28865734968d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab828bb8cf502c343e7fcc65e62a9c98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a9a143e2aca91bf2934bf099d3d267af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.30826058983802795, 0.16620753705501556, 0.3658806383609772, 0.36707037687301636], [0.11164671927690506, 0.36271995306015015, 0.4650300145149231, 0.4107046127319336], [0.07298780232667923, 0.4408380389213562, 0.14397777616977692, 0.15549598634243011], [0.040000829845666885, 0.08228027075529099, 0.292710542678833, 0.036442629992961884], [0.4682316482067108, 0.4430815577507019, 0.376697838306427, 0.30927401781082153]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([5], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f80f151f8cebe35b47c49d940efe0138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8fd2b4b2c2b093d4f7d543ddb32e9d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_404866aa237caae19890499faba0d12a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_729b30425669b695b6a0ac607f255a85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3571544885635376, 0.42178237438201904, 0.09802666306495667, 0.4052501916885376]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_29aa2f2dd9bd0cebeed5ddb89627c0dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_dcc15d397c5efd4fa284b7de61c57271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d3ef2a958d1427c0f615eb2dd1537585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3fc881ca5973ff6e7c4e4da998b8cdfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4320468604564667, 0.3535347282886505, 0.12520934641361237, 0.23588906228542328]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d5bafca3a81d0db71e7a119f20002058(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec35c25bea37ebcc9d1750caf8882e62
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_31eb29313a2f5adf788cb7ef7ebae382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a84e4b50450ac284ca75c853ae8663be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f48e4721edb8f5dcc1bb3dfd9cb777c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f89971896f179f169fbd26c9a499384c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5c691a436e0cf51371ad47d4ce46348c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6253e3a80f16035f6da6589e6b1050e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec35c25bea37ebcc9d1750caf8882e62
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2c6c9620eea857f802de4ac30f79d74e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a84e4b50450ac284ca75c853ae8663be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1dd354175e3fba474e0d439d27fdc240(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f89971896f179f169fbd26c9a499384c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ce1001b41e8feeed4b7ae1ef038bf991(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fedf488a264409407e548188bad1d830
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([100], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4b03e5b65348feb9a43c5df4748d6cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a383e06e16f6eebfe8028e6b10796710
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_167627ed408a304f576f27e5ddc80a1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a649f60367e99c964bbd60f4e35d477
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_362bc4eedfaa62ab44852f437aca7eac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ab828bb8cf502c343e7fcc65e62a9c98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ee575473736bfcbc8099841362dc8cce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.37920650839805603, 0.32739609479904175, 0.01771315187215805, 0.4222439229488373]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([1], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b3393f552644485dcf2be98ef60f4eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec35c25bea37ebcc9d1750caf8882e62
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9388738139bda4017e80174c2d790129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a84e4b50450ac284ca75c853ae8663be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cdb2f81ec029f446c91b8c8534d99bc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f89971896f179f169fbd26c9a499384c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_dd01880c5596f8d545ce676bdb4d603a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.120260089635849, 0.10687462985515594, 0.4020973742008209, 0.1944931298494339], [0.14424557983875275, 0.34658628702163696, 0.2673088014125824, 0.49299418926239014], [0.4701901376247406, 0.2641819417476654, 0.31320127844810486, 0.33311450481414795], [0.037186916917562485, 0.23197132349014282, 0.4214165210723877, 0.09179126471281052], [0.10999499261379242, 0.04785877466201782, 0.09373696148395538, 0.3439233601093292], [0.3809078335762024, 0.11442305147647858, 0.10039916634559631, 0.17913693189620972], [0.41799628734588623, 0.3991064727306366, 0.35630565881729126, 0.16492298245429993]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_982d33a60e3bc08998502884855ce0f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd66776ca8bd45922283de8725111a52
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2b820161a88956f2776a61a4b13394ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15438c4973df542e48bc7e701de9af02
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_32493094d0ec012c8f1a779c54fb08a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f7fe477b49a132218570839021cf8e57
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ebc4ef1059c03ea46009ae7f26b658da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11187709122896194, 0.40898066759109497, 0.25907111167907715, 0.42682981491088867], [0.1378955841064453, 0.0975421741604805, 0.119081050157547, 0.19117246568202972], [0.07604412734508514, 0.02066691406071186, 0.4777871072292328, 0.4768635928630829], [0.4223090708255768, 0.32861530780792236, 0.30099281668663025, 0.1979154497385025], [0.48413437604904175, 0.29007235169410706, 0.49729445576667786, 0.22377794981002808], [0.4088941812515259, 0.1840423196554184, 0.017735710367560387, 0.06378057599067688]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([6], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d59ad5d1f19e30f11e441d5bfbd31bdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec35c25bea37ebcc9d1750caf8882e62
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1081fbeca053a156c5ed60c656824a51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a84e4b50450ac284ca75c853ae8663be
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_6c66cc5a689fe16eaf5c2c89e8b719fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f89971896f179f169fbd26c9a499384c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()