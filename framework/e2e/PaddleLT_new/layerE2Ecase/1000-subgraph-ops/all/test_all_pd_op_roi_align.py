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


    class TestPrimitiveOp_6a6faa1c5ce23fe6c8671569b8c5cd69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02e47bf781c1e34fcde4e2a047af9310
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.28204044699668884, 0.17822766304016113, 0.41775569319725037, 0.033561524003744125], [0.35253238677978516, 0.49377918243408203, 0.2525448501110077, 0.17045120894908905]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_b45d49010a15918f38db5e2c61707070(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.033389102667570114, 0.48712605237960815, 0.42479053139686584, 0.15717636048793793], [0.03449510410428047, 0.49936267733573914, 0.2634246349334717, 0.3026280999183655]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_0404d1151d87e9c342ed02aedd410335(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915ab8504ebc67fd479b8e9c5bf87d67
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.46628230810165405, 0.40333276987075806, 0.3190357983112335, 0.4544869363307953], [0.3820638060569763, 0.11913549154996872, 0.17213769257068634, 0.25780439376831055]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_57cd808830491fb4b1b6b3a721228bb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.02206171490252018, 0.2801748514175415, 0.18513378500938416, 0.06324756890535355], [0.29623863101005554, 0.28500470519065857, 0.46523305773735046, 0.16689153015613556], [0.15229283273220062, 0.05340076982975006, 0.2670149803161621, 0.49961867928504944], [0.07245133072137833, 0.44108298420906067, 0.14271855354309082, 0.1576385349035263], [0.26750293374061584, 0.2523600161075592, 0.4398418962955475, 0.2775665819644928], [0.35968130826950073, 0.04996968433260918, 0.2262144535779953, 0.4627487063407898], [0.31461915373802185, 0.2483227401971817, 0.2315196990966797, 0.2304498553276062]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_8821652c3a4616c1c498d0e2ae24391c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4939478933811188, 0.09468913823366165, 0.4164721369743347, 0.3881565034389496], [0.3569675087928772, 0.3765116333961487, 0.06662866473197937, 0.26502910256385803], [0.48624157905578613, 0.40772998332977295, 0.4239579141139984, 0.16497191786766052], [0.3366274833679199, 0.2810962498188019, 0.1322024017572403, 0.19090597331523895], [0.27363884449005127, 0.2261369675397873, 0.270373672246933, 0.060821935534477234], [0.2501159608364105, 0.40895581245422363, 0.4449658691883087, 0.08937407284975052]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_dca34757c397f4b490250e72681756f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.18688137829303741, 0.31889158487319946, 0.010841316543519497, 0.0022586046252399683], [0.07295744866132736, 0.34488511085510254, 0.41417253017425537, 0.2523491680622101], [0.39163944125175476, 0.019542060792446136, 0.1724236011505127, 0.3455236554145813]], dtype='float32').reshape([3, 4]),
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


    class TestPrimitiveOp_18ba2a8291e4855b9713d48255a1e641(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_915ab8504ebc67fd479b8e9c5bf87d67
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.21205809712409973, 0.15692728757858276, 0.28846707940101624, 0.35022953152656555], [0.33836984634399414, 0.4439485967159271, 0.41524678468704224, 0.3063856363296509]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_3fcd4b0176efc575219d4b0661338d4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.15937396883964539, 0.3468873202800751, 0.09539873152971268, 0.08093781769275665]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_5087131d2ee2ea53e1b5576e729c6882(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33295613527297974, 0.4748709201812744, 0.4807755947113037, 0.4424726963043213], [0.4646690785884857, 0.2766610085964203, 0.009535721503198147, 0.23498447239398956], [0.1327226459980011, 0.43393319845199585, 0.4983842074871063, 0.09926091134548187], [0.4816283881664276, 0.4552164673805237, 0.367679238319397, 0.20860713720321655], [0.4341086447238922, 0.28970199823379517, 0.37916260957717896, 0.19855888187885284], [0.02039922960102558, 0.13990655541419983, 0.4250624477863312, 0.21649450063705444], [0.3156362473964691, 0.4491093158721924, 0.42121651768684387, 0.38926970958709717]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_9d8ca9b314b3812afde741f435e4ae9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.40185317397117615, 0.09948693215847015, 0.018622536212205887, 0.4672163724899292]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_c44040b812a38ac5417f0806c11afba8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.44700753688812256, 0.1607520580291748, 0.31537771224975586, 0.3964860439300537], [0.3145931363105774, 0.048773035407066345, 0.32795286178588867, 0.27992427349090576], [0.12154681980609894, 0.38812240958213806, 0.304490864276886, 0.1556045264005661], [0.06200443580746651, 0.4322132170200348, 0.03444971516728401, 0.23482581973075867], [0.4211438298225403, 0.42408350110054016, 0.14282457530498505, 0.017275415360927582]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_f9b1d54c33ce83ff371a10b3eecd2bc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33940258622169495, 0.04308023676276207, 0.12667907774448395, 0.24631406366825104], [0.23081062734127045, 0.13120730221271515, 0.1732984334230423, 0.08053600043058395], [0.49378523230552673, 0.27325204014778137, 0.4451070725917816, 0.28510376811027527], [0.3046403229236603, 0.36682310700416565, 0.47246965765953064, 0.10411851108074188], [0.0993577241897583, 0.07371293753385544, 0.026188546791672707, 0.13281221687793732], [0.2729603052139282, 0.15208522975444794, 0.04973509907722473, 0.2868596613407135], [0.03855523094534874, 0.3900150954723358, 0.22766464948654175, 0.11040422320365906]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_675d1f259325f08a044b47262fd40499(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3765825927257538, 0.0737718939781189, 0.17367011308670044, 0.13579215109348297], [0.39437875151634216, 0.4508989453315735, 0.4255974292755127, 0.24978524446487427], [0.46405890583992004, 0.2909151613712311, 0.3259059190750122, 0.14486196637153625], [0.039835941046476364, 0.08892933279275894, 0.2858670651912689, 0.43800050020217896], [0.3433256149291992, 0.3984774351119995, 0.33644017577171326, 0.3013238310813904], [0.007092955522239208, 0.0429738350212574, 0.16837380826473236, 0.2850581705570221], [0.17509058117866516, 0.16216276586055756, 0.1184927299618721, 0.2952444553375244]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_81b2030dc7a7ace5e1a9fd18228054d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.15815450251102448, 0.21932286024093628, 0.22791311144828796, 0.08859983086585999]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_b72dc4a2420d9000209173285194faeb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4212192893028259, 0.3294081687927246, 0.14390289783477783, 0.47526413202285767], [0.03612793609499931, 0.11221010982990265, 0.29807817935943604, 0.4925484359264374], [0.12670211493968964, 0.12889350950717926, 0.10977347195148468, 0.30061158537864685], [0.3750476837158203, 0.4990573525428772, 0.3110734522342682, 0.22598418593406677], [0.4775603115558624, 0.06484051048755646, 0.15023624897003174, 0.08388227224349976]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_6f392a3557bdd5b292c0431b76e0b38f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4228264391422272, 0.40261387825012207, 0.0521089993417263, 0.3682134449481964]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_221b7b1ce1274277da4f2102982d629e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.141110360622406, 0.2614602744579315, 0.4805050492286682, 0.29671505093574524]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_ab9e5e6896fa3956f12bb7c25fabc2b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.20489901304244995, 0.30702176690101624, 0.44525042176246643, 0.4422963559627533]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_25e41569f211d9129e7b96c4ae618692(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01d1954dfc87a32b602cd9694037371c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.21851125359535217, 0.13687719404697418, 0.18843640387058258, 0.4617670476436615], [0.04193633422255516, 0.30800360441207886, 0.0410027913749218, 0.11419687420129776], [0.41583868861198425, 0.4192369282245636, 0.32702040672302246, 0.24959872663021088], [0.14772185683250427, 0.1804911345243454, 0.33647412061691284, 0.43272924423217773], [0.39211463928222656, 0.2351677417755127, 0.24655355513095856, 0.443164199590683], [0.4015344977378845, 0.018150970339775085, 0.3292650580406189, 0.3905903697013855], [0.3656322658061981, 0.37607818841934204, 0.24317368865013123, 0.4916781187057495]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_bd4da5b8a96840292e0ca497d5934766(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21aec69e095b4fdc66d7948fd63e92ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.26801797747612, 0.49815842509269714, 0.15705350041389465, 0.35779625177383423], [0.1836244761943817, 0.41449683904647827, 0.08548150956630707, 0.07735142856836319], [0.3767048120498657, 0.17572547495365143, 0.36186933517456055, 0.4755352735519409], [0.34678760170936584, 0.21732518076896667, 0.3460763692855835, 0.4643111228942871], [0.1874842643737793, 0.26683056354522705, 0.18528778851032257, 0.4577597379684448], [0.09015223383903503, 0.4094330370426178, 0.12884452939033508, 0.3893778324127197]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_db89c723e2480e48c2f6c7ab63d8f2a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_41774258b4b7d8f8cfd0123ed306e87d
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.28204044699668884, 0.17822766304016113, 0.41775569319725037, 0.033561524003744125], [0.35253238677978516, 0.49377918243408203, 0.2525448501110077, 0.17045120894908905]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_ba9d88bf59665867b0a21f077043223d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f4ea02cf8c5a0bdbe7cfc47596c71e6
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.033389102667570114, 0.48712605237960815, 0.42479053139686584, 0.15717636048793793], [0.03449510410428047, 0.49936267733573914, 0.2634246349334717, 0.3026280999183655]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_338d2472e7a3e55fd36d90241c1488c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e72fbafecc7a71a26b51ac41608c561
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.46628230810165405, 0.40333276987075806, 0.3190357983112335, 0.4544869363307953], [0.3820638060569763, 0.11913549154996872, 0.17213769257068634, 0.25780439376831055]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_56b066ede0cceed006494cf1fe2390b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_081d0e74c33d435e60ffc3c2cb720a51
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.02206171490252018, 0.2801748514175415, 0.18513378500938416, 0.06324756890535355], [0.29623863101005554, 0.28500470519065857, 0.46523305773735046, 0.16689153015613556], [0.15229283273220062, 0.05340076982975006, 0.2670149803161621, 0.49961867928504944], [0.07245133072137833, 0.44108298420906067, 0.14271855354309082, 0.1576385349035263], [0.26750293374061584, 0.2523600161075592, 0.4398418962955475, 0.2775665819644928], [0.35968130826950073, 0.04996968433260918, 0.2262144535779953, 0.4627487063407898], [0.31461915373802185, 0.2483227401971817, 0.2315196990966797, 0.2304498553276062]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_96ee75acbe52db16e0c4a7186690da11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c842a517c20adfc2f894842689fe4df
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4939478933811188, 0.09468913823366165, 0.4164721369743347, 0.3881565034389496], [0.3569675087928772, 0.3765116333961487, 0.06662866473197937, 0.26502910256385803], [0.48624157905578613, 0.40772998332977295, 0.4239579141139984, 0.16497191786766052], [0.3366274833679199, 0.2810962498188019, 0.1322024017572403, 0.19090597331523895], [0.27363884449005127, 0.2261369675397873, 0.270373672246933, 0.060821935534477234], [0.2501159608364105, 0.40895581245422363, 0.4449658691883087, 0.08937407284975052]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_a6cb23d9e1cbe032bda180da831f1404(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d20704158529c8e76be8812ebdaf66e
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.18688137829303741, 0.31889158487319946, 0.010841316543519497, 0.0022586046252399683], [0.07295744866132736, 0.34488511085510254, 0.41417253017425537, 0.2523491680622101], [0.39163944125175476, 0.019542060792446136, 0.1724236011505127, 0.3455236554145813]], dtype='float32').reshape([3, 4]),
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


    class TestPrimitiveOp_c902a1792f04076de0fd702b3f952fcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e72fbafecc7a71a26b51ac41608c561
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.21205809712409973, 0.15692728757858276, 0.28846707940101624, 0.35022953152656555], [0.33836984634399414, 0.4439485967159271, 0.41524678468704224, 0.3063856363296509]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_49e89ffbb3e877e8c3487724217d544f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0eb783dabfcd8a21f0a79de757eae1b
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.15937396883964539, 0.3468873202800751, 0.09539873152971268, 0.08093781769275665]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_8e0c7334f87f996c2659c2dae8921a83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_990d6e8bf7faa38cc7deeb9ba665afeb
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33295613527297974, 0.4748709201812744, 0.4807755947113037, 0.4424726963043213], [0.4646690785884857, 0.2766610085964203, 0.009535721503198147, 0.23498447239398956], [0.1327226459980011, 0.43393319845199585, 0.4983842074871063, 0.09926091134548187], [0.4816283881664276, 0.4552164673805237, 0.367679238319397, 0.20860713720321655], [0.4341086447238922, 0.28970199823379517, 0.37916260957717896, 0.19855888187885284], [0.02039922960102558, 0.13990655541419983, 0.4250624477863312, 0.21649450063705444], [0.3156362473964691, 0.4491093158721924, 0.42121651768684387, 0.38926970958709717]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_ee2a6e5397b6014eb63d4f84c7362185(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92442a46b544acfbbf212198e940afe9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.40185317397117615, 0.09948693215847015, 0.018622536212205887, 0.4672163724899292]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_77093b00b9e10e8dc310f91c2fcc8481(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9edd85f3c2f8cb1be509a96c605c005
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.44700753688812256, 0.1607520580291748, 0.31537771224975586, 0.3964860439300537], [0.3145931363105774, 0.048773035407066345, 0.32795286178588867, 0.27992427349090576], [0.12154681980609894, 0.38812240958213806, 0.304490864276886, 0.1556045264005661], [0.06200443580746651, 0.4322132170200348, 0.03444971516728401, 0.23482581973075867], [0.4211438298225403, 0.42408350110054016, 0.14282457530498505, 0.017275415360927582]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_f06dbe1daf7dd2c0e90406c82833a5cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1bde784650e08128960e8eb453cff7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33940258622169495, 0.04308023676276207, 0.12667907774448395, 0.24631406366825104], [0.23081062734127045, 0.13120730221271515, 0.1732984334230423, 0.08053600043058395], [0.49378523230552673, 0.27325204014778137, 0.4451070725917816, 0.28510376811027527], [0.3046403229236603, 0.36682310700416565, 0.47246965765953064, 0.10411851108074188], [0.0993577241897583, 0.07371293753385544, 0.026188546791672707, 0.13281221687793732], [0.2729603052139282, 0.15208522975444794, 0.04973509907722473, 0.2868596613407135], [0.03855523094534874, 0.3900150954723358, 0.22766464948654175, 0.11040422320365906]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_71a15ab2124170ac93a578b8a8857297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_081d0e74c33d435e60ffc3c2cb720a51
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3765825927257538, 0.0737718939781189, 0.17367011308670044, 0.13579215109348297], [0.39437875151634216, 0.4508989453315735, 0.4255974292755127, 0.24978524446487427], [0.46405890583992004, 0.2909151613712311, 0.3259059190750122, 0.14486196637153625], [0.039835941046476364, 0.08892933279275894, 0.2858670651912689, 0.43800050020217896], [0.3433256149291992, 0.3984774351119995, 0.33644017577171326, 0.3013238310813904], [0.007092955522239208, 0.0429738350212574, 0.16837380826473236, 0.2850581705570221], [0.17509058117866516, 0.16216276586055756, 0.1184927299618721, 0.2952444553375244]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_f14923d5e49b0f64c3e7deb259b384e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d7ac0b2cb53a2975d08caaad115f5f
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.15815450251102448, 0.21932286024093628, 0.22791311144828796, 0.08859983086585999]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_26d902e3e643a1896c3faf57692d8f16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9edd85f3c2f8cb1be509a96c605c005
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4212192893028259, 0.3294081687927246, 0.14390289783477783, 0.47526413202285767], [0.03612793609499931, 0.11221010982990265, 0.29807817935943604, 0.4925484359264374], [0.12670211493968964, 0.12889350950717926, 0.10977347195148468, 0.30061158537864685], [0.3750476837158203, 0.4990573525428772, 0.3110734522342682, 0.22598418593406677], [0.4775603115558624, 0.06484051048755646, 0.15023624897003174, 0.08388227224349976]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_1955074921a03817b40334e140a8f481(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52909d40df3d7c374d43d932c6d2d06c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4228264391422272, 0.40261387825012207, 0.0521089993417263, 0.3682134449481964]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_4f534b3e6d83a72dbdb78b4002876934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b700f528f4cd736d657f23fac20d8ab2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.141110360622406, 0.2614602744579315, 0.4805050492286682, 0.29671505093574524]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_d1142739f8fdb74252594f74d3538293(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6e63fc367a42a84ff3be03bdcde3162
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.20489901304244995, 0.30702176690101624, 0.44525042176246643, 0.4422963559627533]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_7d66380cd353198166bc175a452d9ba7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1bde784650e08128960e8eb453cff7c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.21851125359535217, 0.13687719404697418, 0.18843640387058258, 0.4617670476436615], [0.04193633422255516, 0.30800360441207886, 0.0410027913749218, 0.11419687420129776], [0.41583868861198425, 0.4192369282245636, 0.32702040672302246, 0.24959872663021088], [0.14772185683250427, 0.1804911345243454, 0.33647412061691284, 0.43272924423217773], [0.39211463928222656, 0.2351677417755127, 0.24655355513095856, 0.443164199590683], [0.4015344977378845, 0.018150970339775085, 0.3292650580406189, 0.3905903697013855], [0.3656322658061981, 0.37607818841934204, 0.24317368865013123, 0.4916781187057495]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_be1320874d241276cdb20ad34bb18e59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6161b6048ce7d7a53519fa392e78cc17
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.26801797747612, 0.49815842509269714, 0.15705350041389465, 0.35779625177383423], [0.1836244761943817, 0.41449683904647827, 0.08548150956630707, 0.07735142856836319], [0.3767048120498657, 0.17572547495365143, 0.36186933517456055, 0.4755352735519409], [0.34678760170936584, 0.21732518076896667, 0.3460763692855835, 0.4643111228942871], [0.1874842643737793, 0.26683056354522705, 0.18528778851032257, 0.4577597379684448], [0.09015223383903503, 0.4094330370426178, 0.12884452939033508, 0.3893778324127197]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_b25d4a686dba26ae8811925315b63859(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.28204044699668884, 0.17822766304016113, 0.41775569319725037, 0.033561524003744125], [0.35253238677978516, 0.49377918243408203, 0.2525448501110077, 0.17045120894908905]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_0ce6ad58a479e06dc7658aeda7da3ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 136, 160], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.033389102667570114, 0.48712605237960815, 0.42479053139686584, 0.15717636048793793], [0.03449510410428047, 0.49936267733573914, 0.2634246349334717, 0.3026280999183655]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_abeb4afa7984f95db91fee9145e79fb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.46628230810165405, 0.40333276987075806, 0.3190357983112335, 0.4544869363307953], [0.3820638060569763, 0.11913549154996872, 0.17213769257068634, 0.25780439376831055]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_d10534786a853a68fd571cd828dd3c06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.02206171490252018, 0.2801748514175415, 0.18513378500938416, 0.06324756890535355], [0.29623863101005554, 0.28500470519065857, 0.46523305773735046, 0.16689153015613556], [0.15229283273220062, 0.05340076982975006, 0.2670149803161621, 0.49961867928504944], [0.07245133072137833, 0.44108298420906067, 0.14271855354309082, 0.1576385349035263], [0.26750293374061584, 0.2523600161075592, 0.4398418962955475, 0.2775665819644928], [0.35968130826950073, 0.04996968433260918, 0.2262144535779953, 0.4627487063407898], [0.31461915373802185, 0.2483227401971817, 0.2315196990966797, 0.2304498553276062]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_7c215f9ff1ef1c538c9c509696632787(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4939478933811188, 0.09468913823366165, 0.4164721369743347, 0.3881565034389496], [0.3569675087928772, 0.3765116333961487, 0.06662866473197937, 0.26502910256385803], [0.48624157905578613, 0.40772998332977295, 0.4239579141139984, 0.16497191786766052], [0.3366274833679199, 0.2810962498188019, 0.1322024017572403, 0.19090597331523895], [0.27363884449005127, 0.2261369675397873, 0.270373672246933, 0.060821935534477234], [0.2501159608364105, 0.40895581245422363, 0.4449658691883087, 0.08937407284975052]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_3b260d1626094899833a638b6ed0fe84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.18688137829303741, 0.31889158487319946, 0.010841316543519497, 0.0022586046252399683], [0.07295744866132736, 0.34488511085510254, 0.41417253017425537, 0.2523491680622101], [0.39163944125175476, 0.019542060792446136, 0.1724236011505127, 0.3455236554145813]], dtype='float32').reshape([3, 4]),
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


    class TestPrimitiveOp_bf0df813cf01e4025ff47b2c115f7a77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.21205809712409973, 0.15692728757858276, 0.28846707940101624, 0.35022953152656555], [0.33836984634399414, 0.4439485967159271, 0.41524678468704224, 0.3063856363296509]], dtype='float32').reshape([2, 4]),
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


    class TestPrimitiveOp_83da0c8b00f1ae9770707d99203e3343(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.15937396883964539, 0.3468873202800751, 0.09539873152971268, 0.08093781769275665]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_0b2788ea270c4824f04abbf613a5d17c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 136, 208], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33295613527297974, 0.4748709201812744, 0.4807755947113037, 0.4424726963043213], [0.4646690785884857, 0.2766610085964203, 0.009535721503198147, 0.23498447239398956], [0.1327226459980011, 0.43393319845199585, 0.4983842074871063, 0.09926091134548187], [0.4816283881664276, 0.4552164673805237, 0.367679238319397, 0.20860713720321655], [0.4341086447238922, 0.28970199823379517, 0.37916260957717896, 0.19855888187885284], [0.02039922960102558, 0.13990655541419983, 0.4250624477863312, 0.21649450063705444], [0.3156362473964691, 0.4491093158721924, 0.42121651768684387, 0.38926970958709717]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_c38cf6cc5e3e3fd0a0927f6bf9d4dfce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.40185317397117615, 0.09948693215847015, 0.018622536212205887, 0.4672163724899292]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_35c98741e5ca9ff56197d3160480fbe1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.44700753688812256, 0.1607520580291748, 0.31537771224975586, 0.3964860439300537], [0.3145931363105774, 0.048773035407066345, 0.32795286178588867, 0.27992427349090576], [0.12154681980609894, 0.38812240958213806, 0.304490864276886, 0.1556045264005661], [0.06200443580746651, 0.4322132170200348, 0.03444971516728401, 0.23482581973075867], [0.4211438298225403, 0.42408350110054016, 0.14282457530498505, 0.017275415360927582]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_7eab88331b348773ebb66c42dbb22a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.33940258622169495, 0.04308023676276207, 0.12667907774448395, 0.24631406366825104], [0.23081062734127045, 0.13120730221271515, 0.1732984334230423, 0.08053600043058395], [0.49378523230552673, 0.27325204014778137, 0.4451070725917816, 0.28510376811027527], [0.3046403229236603, 0.36682310700416565, 0.47246965765953064, 0.10411851108074188], [0.0993577241897583, 0.07371293753385544, 0.026188546791672707, 0.13281221687793732], [0.2729603052139282, 0.15208522975444794, 0.04973509907722473, 0.2868596613407135], [0.03855523094534874, 0.3900150954723358, 0.22766464948654175, 0.11040422320365906]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_710e50b377d8f6cbfc0648f8ba037eae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.3765825927257538, 0.0737718939781189, 0.17367011308670044, 0.13579215109348297], [0.39437875151634216, 0.4508989453315735, 0.4255974292755127, 0.24978524446487427], [0.46405890583992004, 0.2909151613712311, 0.3259059190750122, 0.14486196637153625], [0.039835941046476364, 0.08892933279275894, 0.2858670651912689, 0.43800050020217896], [0.3433256149291992, 0.3984774351119995, 0.33644017577171326, 0.3013238310813904], [0.007092955522239208, 0.0429738350212574, 0.16837380826473236, 0.2850581705570221], [0.17509058117866516, 0.16216276586055756, 0.1184927299618721, 0.2952444553375244]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_609c967c7bf28a6f133023ec666392c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 264], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.15815450251102448, 0.21932286024093628, 0.22791311144828796, 0.08859983086585999]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_469d747b6594eab8091ebead988e2cd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 184, 280], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4212192893028259, 0.3294081687927246, 0.14390289783477783, 0.47526413202285767], [0.03612793609499931, 0.11221010982990265, 0.29807817935943604, 0.4925484359264374], [0.12670211493968964, 0.12889350950717926, 0.10977347195148468, 0.30061158537864685], [0.3750476837158203, 0.4990573525428772, 0.3110734522342682, 0.22598418593406677], [0.4775603115558624, 0.06484051048755646, 0.15023624897003174, 0.08388227224349976]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_359cd25ef1658641228b9ef2eb620a58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 176, 176], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.4228264391422272, 0.40261387825012207, 0.0521089993417263, 0.3682134449481964]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_2277d64fcb9b65c6081ffabdcf9a9cc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 200, 304], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.141110360622406, 0.2614602744579315, 0.4805050492286682, 0.29671505093574524]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_ff0aabb02dbd7ef16d9368c2d4377ad2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.20489901304244995, 0.30702176690101624, 0.44525042176246643, 0.4422963559627533]], dtype='float32').reshape([1, 4]),
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


    class TestPrimitiveOp_606e574574a440ad17f28f7a614e7e63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b98ba588443913377f02b5d8facdaea
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 160, 240], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.21851125359535217, 0.13687719404697418, 0.18843640387058258, 0.4617670476436615], [0.04193633422255516, 0.30800360441207886, 0.0410027913749218, 0.11419687420129776], [0.41583868861198425, 0.4192369282245636, 0.32702040672302246, 0.24959872663021088], [0.14772185683250427, 0.1804911345243454, 0.33647412061691284, 0.43272924423217773], [0.39211463928222656, 0.2351677417755127, 0.24655355513095856, 0.443164199590683], [0.4015344977378845, 0.018150970339775085, 0.3292650580406189, 0.3905903697013855], [0.3656322658061981, 0.37607818841934204, 0.24317368865013123, 0.4916781187057495]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_da0d37800f2087741af82e3f5e4582dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55a63bc3cb6ce40e95463e34d369cc6c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 168, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([[0.26801797747612, 0.49815842509269714, 0.15705350041389465, 0.35779625177383423], [0.1836244761943817, 0.41449683904647827, 0.08548150956630707, 0.07735142856836319], [0.3767048120498657, 0.17572547495365143, 0.36186933517456055, 0.4755352735519409], [0.34678760170936584, 0.21732518076896667, 0.3460763692855835, 0.4643111228942871], [0.1874842643737793, 0.26683056354522705, 0.18528778851032257, 0.4577597379684448], [0.09015223383903503, 0.4094330370426178, 0.12884452939033508, 0.3893778324127197]], dtype='float32').reshape([6, 4]),
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