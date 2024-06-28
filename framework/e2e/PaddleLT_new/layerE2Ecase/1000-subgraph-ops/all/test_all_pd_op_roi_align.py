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


    class TestPrimitiveOp_f581440abf13415f5e3c21b2a9f4a91b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02e47bf781c1e34fcde4e2a047af9310
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.21285614371299744, 0.4041776657104492, 0.35275349020957947, 0.2691345512866974], [0.2065945416688919, 0.23598583042621613, 0.4078265428543091, 0.1987457126379013]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_eaaf0b5a50b044cc465229d2e161dc98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.1861407309770584, 0.31500181555747986, 0.008791757747530937, 0.2193719744682312], [0.4149341285228729, 0.2047792673110962, 0.13938689231872559, 0.09320498257875443]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_4046154ead244e3172d13f7569db16bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915ab8504ebc67fd479b8e9c5bf87d67
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.07871324568986893, 0.2757967412471771, 0.4600844383239746, 0.2323213815689087], [0.19919277727603912, 0.25293204188346863, 0.20971128344535828, 0.23426450788974762]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_8e045a448072cba8419b5a3ce44a0ec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.29310891032218933, 0.06319376081228256, 0.327850341796875, 0.39984428882598877], [0.17362931370735168, 0.2414442002773285, 0.2805016338825226, 0.3628244698047638], [0.08511591702699661, 0.4913937449455261, 0.24752120673656464, 0.05730946362018585], [0.4211045801639557, 0.4457451105117798, 0.4353695511817932, 0.443524569272995], [0.44983530044555664, 0.2074262946844101, 0.4106195569038391, 0.45301324129104614], [0.2362133413553238, 0.22911043465137482, 0.30036306381225586, 0.2968065142631531], [0.28863194584846497, 0.251539021730423, 0.030848540365695953, 0.4922516942024231]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_2832509cfdafd135b7409b2baa7c4693(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.16576938331127167, 0.3791123032569885, 0.4355999827384949, 0.0023606109898537397], [0.46417495608329773, 0.35756731033325195, 0.17070305347442627, 0.1635860800743103], [0.45201581716537476, 0.45891276001930237, 0.2309906929731369, 0.3720448613166809], [0.44901350140571594, 0.2800735533237457, 0.19909116625785828, 0.2635522782802582], [0.17421267926692963, 0.06556755304336548, 0.3417849838733673, 0.34103408455848694], [0.15362094342708588, 0.49990609288215637, 0.005307846702635288, 0.0200535599142313]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_d3998ee1a744401e13fc15778c07241a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.38440650701522827, 0.3501560091972351, 0.353914350271225, 0.13477782905101776], [0.1246904730796814, 0.2588111460208893, 0.3855384290218353, 0.4359793961048126], [0.2229459583759308, 0.22071228921413422, 0.4279545843601227, 0.029600318521261215]], dtype='float32').reshape([3, 4]),
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


    class TestPrimitiveOp_377e470470f042621d04a39fda5e0173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915ab8504ebc67fd479b8e9c5bf87d67
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.43532657623291016, 0.46687716245651245, 0.3278427720069885, 0.10446473956108093], [0.11166555434465408, 0.4263349175453186, 0.11123842746019363, 0.14344440400600433]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_fd3010c31d6605acf180e35053705473(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2238040566444397, 0.011563492938876152, 0.3704405426979065, 0.29472115635871887]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_7c5a1217a78d7c8b02b7a8657207fe5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4067974090576172, 0.22426612675189972, 0.06561402976512909, 0.08147794008255005], [0.1738181710243225, 0.3217422366142273, 0.4032451808452606, 0.24382291734218597], [0.21933263540267944, 0.05699370056390762, 0.3518204987049103, 0.465832382440567], [0.16565661132335663, 0.2725695073604584, 0.014938932843506336, 0.3161398768424988], [0.07797026634216309, 0.07459357380867004, 0.30450886487960815, 0.038236550986766815], [0.26944538950920105, 0.20976261794567108, 0.43011996150016785, 0.3635202646255493], [0.4428406357765198, 0.3498525619506836, 0.2649434506893158, 0.34645187854766846]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_8596f72d8dbe38c2d9af22fdf14bcc97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.07009203732013702, 0.34859538078308105, 0.14423541724681854, 0.4801909625530243]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_eab458621889dc4877c9c5ec48d211e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.17439433932304382, 0.020662004128098488, 0.4366646111011505, 0.2728889286518097], [0.12240549921989441, 0.24261018633842468, 0.2973514199256897, 0.156314879655838], [0.40761297941207886, 0.22659195959568024, 0.07559241354465485, 0.31562525033950806], [0.07620437443256378, 0.27698445320129395, 0.10110228508710861, 0.1968465894460678], [0.2527833878993988, 0.05907091498374939, 0.11603011190891266, 0.041678909212350845]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_8ee840f063fd37dd4933cad699378a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.09072461724281311, 0.24765805900096893, 0.022820264101028442, 0.14492668211460114], [0.27221453189849854, 0.0761546865105629, 0.2743143141269684, 0.11137482523918152], [0.44127634167671204, 0.1320209503173828, 0.18557216227054596, 0.2931707501411438], [0.4452005624771118, 0.35840344429016113, 0.05642806738615036, 0.2801501750946045], [0.41535618901252747, 0.37395140528678894, 0.4865818917751312, 0.49485960602760315], [0.16875706613063812, 0.3940688967704773, 0.03688548505306244, 0.0989731028676033], [0.10508911311626434, 0.3397175073623657, 0.08178718388080597, 0.3122108280658722]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_b88150abd5ca5fffeaa07fb20391a135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.05179205164313316, 0.43257907032966614, 0.45387303829193115, 0.20561820268630981], [0.24862270057201385, 0.024741437286138535, 0.3866288363933563, 0.3768274784088135], [0.23983031511306763, 0.22874388098716736, 0.21488909423351288, 0.47700709104537964], [0.41860711574554443, 0.31253066658973694, 0.3116461932659149, 0.08897895365953445], [0.07988690584897995, 0.13614749908447266, 0.28214341402053833, 0.06329665333032608], [0.4294121563434601, 0.1728123426437378, 0.09979481250047684, 0.1424422264099121], [0.29977691173553467, 0.11897017061710358, 0.21439331769943237, 0.04979977756738663]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_0e277cfa4548d04b84690e9a444c1a00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11988484859466553, 0.2022089809179306, 0.4782188832759857, 0.1624918431043625]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_146bf3d4531097c44aae126196a63393(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3686906397342682, 0.4787021577358246, 0.48923176527023315, 0.47997841238975525], [0.34543925523757935, 0.2998761832714081, 0.2777336835861206, 0.2967449724674225], [0.28960302472114563, 0.46289288997650146, 0.46897920966148376, 0.12904617190361023], [0.35490840673446655, 0.054480522871017456, 0.44187280535697937, 0.4684281051158905], [0.4462490975856781, 0.1875501573085785, 0.4604361951351166, 0.1252194494009018]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_802b6fcba130cf5b9be9847a2d55602f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2667802572250366, 0.30039694905281067, 0.2922790050506592, 0.4462341070175171]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_68e0c561d8e2fb4735d8d7f0e3ccd7ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2822648584842682, 0.3874063193798065, 0.27121636271476746, 0.017748380079865456]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_be6aad5b8de26fb824a0629665ea0319(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33800655603408813, 0.03372206538915634, 0.1964319348335266, 0.15350186824798584]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_6f73df12412d2005b61697bc0bb3a9d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3438580632209778, 0.10636638849973679, 0.028000717982649803, 0.40668386220932007], [0.4063986539840698, 0.353657603263855, 0.10987109690904617, 0.49948033690452576], [0.4550943374633789, 0.3116804361343384, 0.22233928740024567, 0.30103179812431335], [0.47463342547416687, 0.42661070823669434, 0.13912659883499146, 0.3127990961074829], [0.010966498404741287, 0.2572706639766693, 0.46365809440612793, 0.2664984464645386], [0.4284725785255432, 0.36333051323890686, 0.327169805765152, 0.0628637820482254], [0.1629575490951538, 0.1191929504275322, 0.28322362899780273, 0.06393806636333466]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_a65cca55dee7f7fff207d5af96be549c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.32035887241363525, 0.1021905392408371, 0.16249209642410278, 0.15289515256881714], [0.4252414107322693, 0.34034648537635803, 0.03465407341718674, 0.05285079404711723], [0.10380061715841293, 0.22995224595069885, 0.2864528298377991, 0.3551819920539856], [0.3726018965244293, 0.25241681933403015, 0.043521638959646225, 0.25579598546028137], [0.11622543632984161, 0.01757095940411091, 0.4814646244049072, 0.0938255712389946], [0.2863723635673523, 0.15931518375873566, 0.19397640228271484, 0.30772125720977783]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_0387ed39bdbcaec7a2fff4c929b0c6fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41774258b4b7d8f8cfd0123ed306e87d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.21285614371299744, 0.4041776657104492, 0.35275349020957947, 0.2691345512866974], [0.2065945416688919, 0.23598583042621613, 0.4078265428543091, 0.1987457126379013]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_cbd0270644b376c706e7bee8ec9557f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f4ea02cf8c5a0bdbe7cfc47596c71e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.1861407309770584, 0.31500181555747986, 0.008791757747530937, 0.2193719744682312], [0.4149341285228729, 0.2047792673110962, 0.13938689231872559, 0.09320498257875443]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_e36708f1377bd2f7a0a926dca11418f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e72fbafecc7a71a26b51ac41608c561
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.07871324568986893, 0.2757967412471771, 0.4600844383239746, 0.2323213815689087], [0.19919277727603912, 0.25293204188346863, 0.20971128344535828, 0.23426450788974762]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_ea9c5c1c68ae70b1f87d8ac406f274f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_081d0e74c33d435e60ffc3c2cb720a51
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.29310891032218933, 0.06319376081228256, 0.327850341796875, 0.39984428882598877], [0.17362931370735168, 0.2414442002773285, 0.2805016338825226, 0.3628244698047638], [0.08511591702699661, 0.4913937449455261, 0.24752120673656464, 0.05730946362018585], [0.4211045801639557, 0.4457451105117798, 0.4353695511817932, 0.443524569272995], [0.44983530044555664, 0.2074262946844101, 0.4106195569038391, 0.45301324129104614], [0.2362133413553238, 0.22911043465137482, 0.30036306381225586, 0.2968065142631531], [0.28863194584846497, 0.251539021730423, 0.030848540365695953, 0.4922516942024231]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_82293fa3aff8fbf549314b5dab25d222(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c842a517c20adfc2f894842689fe4df
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.16576938331127167, 0.3791123032569885, 0.4355999827384949, 0.0023606109898537397], [0.46417495608329773, 0.35756731033325195, 0.17070305347442627, 0.1635860800743103], [0.45201581716537476, 0.45891276001930237, 0.2309906929731369, 0.3720448613166809], [0.44901350140571594, 0.2800735533237457, 0.19909116625785828, 0.2635522782802582], [0.17421267926692963, 0.06556755304336548, 0.3417849838733673, 0.34103408455848694], [0.15362094342708588, 0.49990609288215637, 0.005307846702635288, 0.0200535599142313]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_580a4748473dd5efc997cb1f34655642(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d20704158529c8e76be8812ebdaf66e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.38440650701522827, 0.3501560091972351, 0.353914350271225, 0.13477782905101776], [0.1246904730796814, 0.2588111460208893, 0.3855384290218353, 0.4359793961048126], [0.2229459583759308, 0.22071228921413422, 0.4279545843601227, 0.029600318521261215]], dtype='float32').reshape([3, 4]),
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


    class TestPrimitiveOp_3db8c5e1e233b11a8444859a48a15124(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e72fbafecc7a71a26b51ac41608c561
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.43532657623291016, 0.46687716245651245, 0.3278427720069885, 0.10446473956108093], [0.11166555434465408, 0.4263349175453186, 0.11123842746019363, 0.14344440400600433]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_273cf8fcb60adc3efb2c092d3f51e88b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0eb783dabfcd8a21f0a79de757eae1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2238040566444397, 0.011563492938876152, 0.3704405426979065, 0.29472115635871887]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_9a53dd8ec98813760130c8697c080fcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_990d6e8bf7faa38cc7deeb9ba665afeb
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4067974090576172, 0.22426612675189972, 0.06561402976512909, 0.08147794008255005], [0.1738181710243225, 0.3217422366142273, 0.4032451808452606, 0.24382291734218597], [0.21933263540267944, 0.05699370056390762, 0.3518204987049103, 0.465832382440567], [0.16565661132335663, 0.2725695073604584, 0.014938932843506336, 0.3161398768424988], [0.07797026634216309, 0.07459357380867004, 0.30450886487960815, 0.038236550986766815], [0.26944538950920105, 0.20976261794567108, 0.43011996150016785, 0.3635202646255493], [0.4428406357765198, 0.3498525619506836, 0.2649434506893158, 0.34645187854766846]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_6da0279aac09b3779dfeb0ef7e6a90ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92442a46b544acfbbf212198e940afe9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.07009203732013702, 0.34859538078308105, 0.14423541724681854, 0.4801909625530243]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_c67fbd1cdec1921f0cce7281e1d6abc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9edd85f3c2f8cb1be509a96c605c005
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.17439433932304382, 0.020662004128098488, 0.4366646111011505, 0.2728889286518097], [0.12240549921989441, 0.24261018633842468, 0.2973514199256897, 0.156314879655838], [0.40761297941207886, 0.22659195959568024, 0.07559241354465485, 0.31562525033950806], [0.07620437443256378, 0.27698445320129395, 0.10110228508710861, 0.1968465894460678], [0.2527833878993988, 0.05907091498374939, 0.11603011190891266, 0.041678909212350845]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_943790153344514170df0bc9429f2a95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1bde784650e08128960e8eb453cff7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.09072461724281311, 0.24765805900096893, 0.022820264101028442, 0.14492668211460114], [0.27221453189849854, 0.0761546865105629, 0.2743143141269684, 0.11137482523918152], [0.44127634167671204, 0.1320209503173828, 0.18557216227054596, 0.2931707501411438], [0.4452005624771118, 0.35840344429016113, 0.05642806738615036, 0.2801501750946045], [0.41535618901252747, 0.37395140528678894, 0.4865818917751312, 0.49485960602760315], [0.16875706613063812, 0.3940688967704773, 0.03688548505306244, 0.0989731028676033], [0.10508911311626434, 0.3397175073623657, 0.08178718388080597, 0.3122108280658722]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_f0aafbeb0a1fbcb8fc63442b31076b24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_081d0e74c33d435e60ffc3c2cb720a51
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.05179205164313316, 0.43257907032966614, 0.45387303829193115, 0.20561820268630981], [0.24862270057201385, 0.024741437286138535, 0.3866288363933563, 0.3768274784088135], [0.23983031511306763, 0.22874388098716736, 0.21488909423351288, 0.47700709104537964], [0.41860711574554443, 0.31253066658973694, 0.3116461932659149, 0.08897895365953445], [0.07988690584897995, 0.13614749908447266, 0.28214341402053833, 0.06329665333032608], [0.4294121563434601, 0.1728123426437378, 0.09979481250047684, 0.1424422264099121], [0.29977691173553467, 0.11897017061710358, 0.21439331769943237, 0.04979977756738663]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_0235ec9c57e075c5964a142c6dd4eaed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d7ac0b2cb53a2975d08caaad115f5f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11988484859466553, 0.2022089809179306, 0.4782188832759857, 0.1624918431043625]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_72c7e73f50f1a87c1cc6abf38347f8c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9edd85f3c2f8cb1be509a96c605c005
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3686906397342682, 0.4787021577358246, 0.48923176527023315, 0.47997841238975525], [0.34543925523757935, 0.2998761832714081, 0.2777336835861206, 0.2967449724674225], [0.28960302472114563, 0.46289288997650146, 0.46897920966148376, 0.12904617190361023], [0.35490840673446655, 0.054480522871017456, 0.44187280535697937, 0.4684281051158905], [0.4462490975856781, 0.1875501573085785, 0.4604361951351166, 0.1252194494009018]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_bcb37f1261b01cc6191e6c12adb918dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52909d40df3d7c374d43d932c6d2d06c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2667802572250366, 0.30039694905281067, 0.2922790050506592, 0.4462341070175171]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_47bc0a80a433175e13d7a02f99f8744f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b700f528f4cd736d657f23fac20d8ab2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2822648584842682, 0.3874063193798065, 0.27121636271476746, 0.017748380079865456]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_97e04520c7e833c1c069846c42e67342(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e63fc367a42a84ff3be03bdcde3162
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33800655603408813, 0.03372206538915634, 0.1964319348335266, 0.15350186824798584]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_7afd4f3616444013e3b52c1a9a65e887(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1bde784650e08128960e8eb453cff7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3438580632209778, 0.10636638849973679, 0.028000717982649803, 0.40668386220932007], [0.4063986539840698, 0.353657603263855, 0.10987109690904617, 0.49948033690452576], [0.4550943374633789, 0.3116804361343384, 0.22233928740024567, 0.30103179812431335], [0.47463342547416687, 0.42661070823669434, 0.13912659883499146, 0.3127990961074829], [0.010966498404741287, 0.2572706639766693, 0.46365809440612793, 0.2664984464645386], [0.4284725785255432, 0.36333051323890686, 0.327169805765152, 0.0628637820482254], [0.1629575490951538, 0.1191929504275322, 0.28322362899780273, 0.06393806636333466]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_80a500f3aa481451facc3a19ad6f1848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6161b6048ce7d7a53519fa392e78cc17
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.32035887241363525, 0.1021905392408371, 0.16249209642410278, 0.15289515256881714], [0.4252414107322693, 0.34034648537635803, 0.03465407341718674, 0.05285079404711723], [0.10380061715841293, 0.22995224595069885, 0.2864528298377991, 0.3551819920539856], [0.3726018965244293, 0.25241681933403015, 0.043521638959646225, 0.25579598546028137], [0.11622543632984161, 0.01757095940411091, 0.4814646244049072, 0.0938255712389946], [0.2863723635673523, 0.15931518375873566, 0.19397640228271484, 0.30772125720977783]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_c0d95dd5c7bba6827d2b88e212229a07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.21285614371299744, 0.4041776657104492, 0.35275349020957947, 0.2691345512866974], [0.2065945416688919, 0.23598583042621613, 0.4078265428543091, 0.1987457126379013]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_50c7900a34a662ca72b385e01e177065(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.1861407309770584, 0.31500181555747986, 0.008791757747530937, 0.2193719744682312], [0.4149341285228729, 0.2047792673110962, 0.13938689231872559, 0.09320498257875443]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_5df9f4bcdab1ad5c7b46a02e19815875(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.07871324568986893, 0.2757967412471771, 0.4600844383239746, 0.2323213815689087], [0.19919277727603912, 0.25293204188346863, 0.20971128344535828, 0.23426450788974762]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_0edbab8aa80d2f0adecbdad0d1cd17c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.29310891032218933, 0.06319376081228256, 0.327850341796875, 0.39984428882598877], [0.17362931370735168, 0.2414442002773285, 0.2805016338825226, 0.3628244698047638], [0.08511591702699661, 0.4913937449455261, 0.24752120673656464, 0.05730946362018585], [0.4211045801639557, 0.4457451105117798, 0.4353695511817932, 0.443524569272995], [0.44983530044555664, 0.2074262946844101, 0.4106195569038391, 0.45301324129104614], [0.2362133413553238, 0.22911043465137482, 0.30036306381225586, 0.2968065142631531], [0.28863194584846497, 0.251539021730423, 0.030848540365695953, 0.4922516942024231]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_aa5082155f8a99eacc2b7766c564a720(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.16576938331127167, 0.3791123032569885, 0.4355999827384949, 0.0023606109898537397], [0.46417495608329773, 0.35756731033325195, 0.17070305347442627, 0.1635860800743103], [0.45201581716537476, 0.45891276001930237, 0.2309906929731369, 0.3720448613166809], [0.44901350140571594, 0.2800735533237457, 0.19909116625785828, 0.2635522782802582], [0.17421267926692963, 0.06556755304336548, 0.3417849838733673, 0.34103408455848694], [0.15362094342708588, 0.49990609288215637, 0.005307846702635288, 0.0200535599142313]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_78128bd81595737490636f4bdf262b73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.38440650701522827, 0.3501560091972351, 0.353914350271225, 0.13477782905101776], [0.1246904730796814, 0.2588111460208893, 0.3855384290218353, 0.4359793961048126], [0.2229459583759308, 0.22071228921413422, 0.4279545843601227, 0.029600318521261215]], dtype='float32').reshape([3, 4]),
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


    class TestPrimitiveOp_d997b9e4ff395e4b604cb3fc08e36b16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.43532657623291016, 0.46687716245651245, 0.3278427720069885, 0.10446473956108093], [0.11166555434465408, 0.4263349175453186, 0.11123842746019363, 0.14344440400600433]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_bf791ae9a964d31c54c3e942361f495d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2238040566444397, 0.011563492938876152, 0.3704405426979065, 0.29472115635871887]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_ab0f2c2e278275d49f2f614589e462aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4067974090576172, 0.22426612675189972, 0.06561402976512909, 0.08147794008255005], [0.1738181710243225, 0.3217422366142273, 0.4032451808452606, 0.24382291734218597], [0.21933263540267944, 0.05699370056390762, 0.3518204987049103, 0.465832382440567], [0.16565661132335663, 0.2725695073604584, 0.014938932843506336, 0.3161398768424988], [0.07797026634216309, 0.07459357380867004, 0.30450886487960815, 0.038236550986766815], [0.26944538950920105, 0.20976261794567108, 0.43011996150016785, 0.3635202646255493], [0.4428406357765198, 0.3498525619506836, 0.2649434506893158, 0.34645187854766846]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_0597a4dbc62992518b94035b3bc69d86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.07009203732013702, 0.34859538078308105, 0.14423541724681854, 0.4801909625530243]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_6d5d19dfed68c8a80138377e63c4a752(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.17439433932304382, 0.020662004128098488, 0.4366646111011505, 0.2728889286518097], [0.12240549921989441, 0.24261018633842468, 0.2973514199256897, 0.156314879655838], [0.40761297941207886, 0.22659195959568024, 0.07559241354465485, 0.31562525033950806], [0.07620437443256378, 0.27698445320129395, 0.10110228508710861, 0.1968465894460678], [0.2527833878993988, 0.05907091498374939, 0.11603011190891266, 0.041678909212350845]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_4f1447faf9b24d0950ad0864b8f735c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.09072461724281311, 0.24765805900096893, 0.022820264101028442, 0.14492668211460114], [0.27221453189849854, 0.0761546865105629, 0.2743143141269684, 0.11137482523918152], [0.44127634167671204, 0.1320209503173828, 0.18557216227054596, 0.2931707501411438], [0.4452005624771118, 0.35840344429016113, 0.05642806738615036, 0.2801501750946045], [0.41535618901252747, 0.37395140528678894, 0.4865818917751312, 0.49485960602760315], [0.16875706613063812, 0.3940688967704773, 0.03688548505306244, 0.0989731028676033], [0.10508911311626434, 0.3397175073623657, 0.08178718388080597, 0.3122108280658722]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_96b2fd07f60d2bb46fb6b37466f53001(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.05179205164313316, 0.43257907032966614, 0.45387303829193115, 0.20561820268630981], [0.24862270057201385, 0.024741437286138535, 0.3866288363933563, 0.3768274784088135], [0.23983031511306763, 0.22874388098716736, 0.21488909423351288, 0.47700709104537964], [0.41860711574554443, 0.31253066658973694, 0.3116461932659149, 0.08897895365953445], [0.07988690584897995, 0.13614749908447266, 0.28214341402053833, 0.06329665333032608], [0.4294121563434601, 0.1728123426437378, 0.09979481250047684, 0.1424422264099121], [0.29977691173553467, 0.11897017061710358, 0.21439331769943237, 0.04979977756738663]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_298f38df392a9be5a33ede8b9694bc6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.11988484859466553, 0.2022089809179306, 0.4782188832759857, 0.1624918431043625]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_448e208d16aec2af2999d98b9ba0cc78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3686906397342682, 0.4787021577358246, 0.48923176527023315, 0.47997841238975525], [0.34543925523757935, 0.2998761832714081, 0.2777336835861206, 0.2967449724674225], [0.28960302472114563, 0.46289288997650146, 0.46897920966148376, 0.12904617190361023], [0.35490840673446655, 0.054480522871017456, 0.44187280535697937, 0.4684281051158905], [0.4462490975856781, 0.1875501573085785, 0.4604361951351166, 0.1252194494009018]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_db6304a774f8fe4248965127791f071c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2667802572250366, 0.30039694905281067, 0.2922790050506592, 0.4462341070175171]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_c569b387b5f75cf98a96b0d39ead52db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.2822648584842682, 0.3874063193798065, 0.27121636271476746, 0.017748380079865456]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_5bac8a48865cbc8b8658583ec84c32d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33800655603408813, 0.03372206538915634, 0.1964319348335266, 0.15350186824798584]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_8ccf3e4f78a032ae5414eb289bc87027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3438580632209778, 0.10636638849973679, 0.028000717982649803, 0.40668386220932007], [0.4063986539840698, 0.353657603263855, 0.10987109690904617, 0.49948033690452576], [0.4550943374633789, 0.3116804361343384, 0.22233928740024567, 0.30103179812431335], [0.47463342547416687, 0.42661070823669434, 0.13912659883499146, 0.3127990961074829], [0.010966498404741287, 0.2572706639766693, 0.46365809440612793, 0.2664984464645386], [0.4284725785255432, 0.36333051323890686, 0.327169805765152, 0.0628637820482254], [0.1629575490951538, 0.1191929504275322, 0.28322362899780273, 0.06393806636333466]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_51a04d9d1f5c2e5876d8b8ff0b5d5f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.32035887241363525, 0.1021905392408371, 0.16249209642410278, 0.15289515256881714], [0.4252414107322693, 0.34034648537635803, 0.03465407341718674, 0.05285079404711723], [0.10380061715841293, 0.22995224595069885, 0.2864528298377991, 0.3551819920539856], [0.3726018965244293, 0.25241681933403015, 0.043521638959646225, 0.25579598546028137], [0.11622543632984161, 0.01757095940411091, 0.4814646244049072, 0.0938255712389946], [0.2863723635673523, 0.15931518375873566, 0.19397640228271484, 0.30772125720977783]], dtype='float32').reshape([6, 4]),
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