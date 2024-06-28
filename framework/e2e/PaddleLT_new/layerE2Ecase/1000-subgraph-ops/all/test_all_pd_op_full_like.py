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
    class PrimitiveOp_d243c670e466c4d88a304f47c2f13366(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_22ac846d27f3ee90921d541b4f14723d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d243c670e466c4d88a304f47c2f13366
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3174a6012b9b8938d5e7b98441e586b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d243c670e466c4d88a304f47c2f13366
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3174a6012b9b8938d5e7b98441e586b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d243c670e466c4d88a304f47c2f13366
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_f34329015224d81c946c39aeef818538(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.bool, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1812df288523c97ab858bdf0bc8d09f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f34329015224d81c946c39aeef818538
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_2b56abf12a84e63aa9379a5faf0a9922(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.float32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3535e7fce8d9ee19f56a08a3d48c646(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b56abf12a84e63aa9379a5faf0a9922
        def get_inputs(self):
            return [
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_0f8aa0bb8cc440496feed24b457f48c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c75e89f3c3a171d3c9ff62d5dd57351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f8aa0bb8cc440496feed24b457f48c3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9c75e89f3c3a171d3c9ff62d5dd57351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f8aa0bb8cc440496feed24b457f48c3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_420f25d00af76e0ac62ee0b3daa58184(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f8aa0bb8cc440496feed24b457f48c3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_420f25d00af76e0ac62ee0b3daa58184(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f8aa0bb8cc440496feed24b457f48c3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_393c4a59f84af6bb2ac5390fa482f9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f8aa0bb8cc440496feed24b457f48c3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_393c4a59f84af6bb2ac5390fa482f9d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f8aa0bb8cc440496feed24b457f48c3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_e81e6e03b103ea0c40ba563773f545d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4efa5ecf685899325664792da381fe55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e81e6e03b103ea0c40ba563773f545d7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6c2ec23dffb5f4e0452d9c1a905f17af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e81e6e03b103ea0c40ba563773f545d7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6c2ec23dffb5f4e0452d9c1a905f17af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e81e6e03b103ea0c40ba563773f545d7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e85cf5c2bd6e561bb742e64d6cc70efd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f34329015224d81c946c39aeef818538
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_49b3f09453e4bc0400008b070e71ca4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68a5ced3c898e1b19767c9f3b3389cc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b3f09453e4bc0400008b070e71ca4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_049f0d5062c9a47895ba2eac76c9bc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b3f09453e4bc0400008b070e71ca4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_049f0d5062c9a47895ba2eac76c9bc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b3f09453e4bc0400008b070e71ca4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_01198f64c3eae3e5a68f5a1b128b1375(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f34329015224d81c946c39aeef818538
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c9effa9e4426bbd1f1f9ccfd7f9cb4db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f8aa0bb8cc440496feed24b457f48c3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c9effa9e4426bbd1f1f9ccfd7f9cb4db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f8aa0bb8cc440496feed24b457f48c3
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_22ac846d27f3ee90921d541b4f14723d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d243c670e466c4d88a304f47c2f13366
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3174a6012b9b8938d5e7b98441e586b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d243c670e466c4d88a304f47c2f13366
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3174a6012b9b8938d5e7b98441e586b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d243c670e466c4d88a304f47c2f13366
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9c0d94940a75d05d8dc96fa7024d3f18(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.bool, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2e9884fe27b751bb703f3ba9193f6691(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c0d94940a75d05d8dc96fa7024d3f18
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f3535e7fce8d9ee19f56a08a3d48c646(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b56abf12a84e63aa9379a5faf0a9922
        def get_inputs(self):
            return [
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_9d4058f15ec7f7d1e2782f772a2255b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2002], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42cb28dc0c228fc4bca6941638c13fdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d4058f15ec7f7d1e2782f772a2255b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_42cb28dc0c228fc4bca6941638c13fdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d4058f15ec7f7d1e2782f772a2255b8
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_376c90828ba0db0e63e473382e1045e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1021], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60f4fe0d3b8142a983d5ead850f8dfbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_376c90828ba0db0e63e473382e1045e7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_60f4fe0d3b8142a983d5ead850f8dfbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_376c90828ba0db0e63e473382e1045e7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_33d9a7e2a41dfc05596b0bc56d097077(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1002], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59131d11f91eb7cf2b9ef3cfe104d5a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33d9a7e2a41dfc05596b0bc56d097077
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_59131d11f91eb7cf2b9ef3cfe104d5a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33d9a7e2a41dfc05596b0bc56d097077
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4efa5ecf685899325664792da381fe55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e81e6e03b103ea0c40ba563773f545d7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6c2ec23dffb5f4e0452d9c1a905f17af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e81e6e03b103ea0c40ba563773f545d7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6c2ec23dffb5f4e0452d9c1a905f17af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e81e6e03b103ea0c40ba563773f545d7
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_f05c1b305009c90ef157f4c2444f2203(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.bool, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bcf91e956dc57af0101a55c4dabbb294(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f05c1b305009c90ef157f4c2444f2203
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_68a5ced3c898e1b19767c9f3b3389cc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b3f09453e4bc0400008b070e71ca4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_049f0d5062c9a47895ba2eac76c9bc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b3f09453e4bc0400008b070e71ca4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_049f0d5062c9a47895ba2eac76c9bc5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b3f09453e4bc0400008b070e71ca4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_24f38beb2c96ecaff4375102340581ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.bool, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='bool'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05d928e28132c5ae4c07d2d96e4ce64b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24f38beb2c96ecaff4375102340581ee
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_389cf5046bbbe59e85648627aff25fcb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1027], dtype='int32'),
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_413b6f032fcc7557b58fac5c23405676(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_389cf5046bbbe59e85648627aff25fcb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_413b6f032fcc7557b58fac5c23405676(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_389cf5046bbbe59e85648627aff25fcb
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_26647256fb2028e534f3b742c07b6e4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98ba47e4fac9ec45612aac05c84b0938(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26647256fb2028e534f3b742c07b6e4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d3d610b612dd9f2b652f9d56083c6b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26647256fb2028e534f3b742c07b6e4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0d3d610b612dd9f2b652f9d56083c6b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26647256fb2028e534f3b742c07b6e4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_780072d3a93141a4fa830466ba41145e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.bool, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2fb36b240c4b13545c187a4f9c3ff1f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_780072d3a93141a4fa830466ba41145e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_8566bb452c91ba38e017e8cdbf333588(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.float32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_061b0d384b0c9bfe56b331a3f936fb9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8566bb452c91ba38e017e8cdbf333588
        def get_inputs(self):
            return [
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_8524fcfa31a18f1b61033eaf8f339e5b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle._C_ops.full_like(input_0, input_1, paddle.int32, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_beb41f0e7c18bc5c6de8890589c26452(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8524fcfa31a18f1b61033eaf8f339e5b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_beb41f0e7c18bc5c6de8890589c26452(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8524fcfa31a18f1b61033eaf8f339e5b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e7210cf8d48d23e9f8f88a2e903bf04b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8524fcfa31a18f1b61033eaf8f339e5b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e7210cf8d48d23e9f8f88a2e903bf04b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8524fcfa31a18f1b61033eaf8f339e5b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1021], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_946c3852c602996bdcb2fed8f85bc815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8524fcfa31a18f1b61033eaf8f339e5b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_946c3852c602996bdcb2fed8f85bc815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8524fcfa31a18f1b61033eaf8f339e5b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1002], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_06039180b95af2da099e9fb8832f3f4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26647256fb2028e534f3b742c07b6e4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([80.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7a6773371afb07c1b373eaa653c1cfc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26647256fb2028e534f3b742c07b6e4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7a6773371afb07c1b373eaa653c1cfc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26647256fb2028e534f3b742c07b6e4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7905b2e8acb7573befd4a8d0fe81ea16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_780072d3a93141a4fa830466ba41145e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e30c53a6a5f589c962e8ce29aea0185a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26647256fb2028e534f3b742c07b6e4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([20.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44b187f123439285cb9031b0f5953c9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26647256fb2028e534f3b742c07b6e4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_44b187f123439285cb9031b0f5953c9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26647256fb2028e534f3b742c07b6e4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8ca8dd2d486574904f8575e6ad3c2280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_780072d3a93141a4fa830466ba41145e
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
                paddle.to_tensor([0.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_65a4543ccdcabb16283bd8a7a8b7467d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8524fcfa31a18f1b61033eaf8f339e5b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_65a4543ccdcabb16283bd8a7a8b7467d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8524fcfa31a18f1b61033eaf8f339e5b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1027], dtype='int32'),
                paddle.to_tensor([1.0], dtype='float32').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()