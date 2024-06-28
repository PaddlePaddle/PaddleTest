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
    class PrimitiveOp_bb19115e604275b814c3e9625c3fde01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='bool'),
                paddle.static.InputSpec(shape=[None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_797af715e3943298ba8996c984a17faa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a91be7225c04a25d6f3d6044984910a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_275c8bd09215ef015e7643987cdd1a70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_797af715e3943298ba8996c984a17faa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ba36015ebe878925e036fc6fbc4bbe31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ba36015ebe878925e036fc6fbc4bbe31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dde1c184b64f1045acc9fc57c2f44e69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_be6c87ea3905a3cb742a939a31e9c133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a364961509cc9ae9c89a4fec3d9ce844(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c420767e2c0aee96602b571bfe8f0e3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_93b214b6e6e05add8fc4940df95c3aa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_150ede426e9e7ee129b7205340691eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c420767e2c0aee96602b571bfe8f0e3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_797af715e3943298ba8996c984a17faa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_93b214b6e6e05add8fc4940df95c3aa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_be6c87ea3905a3cb742a939a31e9c133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_a0060a254594617d334a54ffe33628b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800], dtype='bool'),
                paddle.static.InputSpec(shape=[3800], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e98b5bec5553d67aa27fa8299c67b31c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0060a254594617d334a54ffe33628b3
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_29b879a5a1a27431be855a545a1df8dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150], dtype='bool'),
                paddle.static.InputSpec(shape=[150], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0ec77b587e0995fc529e11d513ea603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29b879a5a1a27431be855a545a1df8dc
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_39b4f60d8f2e7053c0eb6496dfd56b47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='bool'),
                paddle.static.InputSpec(shape=[40], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03c8eaf02a4322d682256587c044b641(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39b4f60d8f2e7053c0eb6496dfd56b47
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e98b5bec5553d67aa27fa8299c67b31c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0060a254594617d334a54ffe33628b3
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_754aa58cc606ba00d505ced671dd52ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200], dtype='bool'),
                paddle.static.InputSpec(shape=[15200], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6fd2257ce76542b0f5db2cdcddeb27a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_754aa58cc606ba00d505ced671dd52ac
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b6fd2257ce76542b0f5db2cdcddeb27a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_754aa58cc606ba00d505ced671dd52ac
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_5080692de75dd8be0bb9711ae8037550(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2204], dtype='bool'),
                paddle.static.InputSpec(shape=[2204], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8843e8b22cfe806e6834f629f43bb38b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5080692de75dd8be0bb9711ae8037550
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_0939fbd5cc4766e8fa25e762b3509f92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70], dtype='bool'),
                paddle.static.InputSpec(shape=[70], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a2447e3dd42fe5d1c9f5f0801551ce9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0939fbd5cc4766e8fa25e762b3509f92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_0fb2eb2a1426c36fc468146281cda45a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[551], dtype='bool'),
                paddle.static.InputSpec(shape=[551], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c328dde0170df9af537cfa29d924c2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fb2eb2a1426c36fc468146281cda45a
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_8033144ec6ef8e29e53a6183dd257a96(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247], dtype='bool'),
                paddle.static.InputSpec(shape=[247], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9ef2177b0c93371fd3036b92781e02c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8033144ec6ef8e29e53a6183dd257a96
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_71528a90e17f7562ed357b2bcc05e679(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950], dtype='bool'),
                paddle.static.InputSpec(shape=[950], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e763956296edddfbdf9c2b5996a07e91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71528a90e17f7562ed357b2bcc05e679
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_53b5ef3e271ab058940750902b30572f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return paddle.logical_and(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8816], dtype='bool'),
                paddle.static.InputSpec(shape=[8816], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e4fce07b5e6fd951c73ead2abbe0c6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53b5ef3e271ab058940750902b30572f
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b9ef2177b0c93371fd3036b92781e02c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8033144ec6ef8e29e53a6183dd257a96
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e98b5bec5553d67aa27fa8299c67b31c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a0060a254594617d334a54ffe33628b3
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e763956296edddfbdf9c2b5996a07e91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71528a90e17f7562ed357b2bcc05e679
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3a2447e3dd42fe5d1c9f5f0801551ce9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0939fbd5cc4766e8fa25e762b3509f92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_797af715e3943298ba8996c984a17faa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a91be7225c04a25d6f3d6044984910a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[150], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_275c8bd09215ef015e7643987cdd1a70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[40], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_797af715e3943298ba8996c984a17faa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ba36015ebe878925e036fc6fbc4bbe31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ba36015ebe878925e036fc6fbc4bbe31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[15200], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dde1c184b64f1045acc9fc57c2f44e69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[2204], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_be6c87ea3905a3cb742a939a31e9c133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a364961509cc9ae9c89a4fec3d9ce844(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[551], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c420767e2c0aee96602b571bfe8f0e3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_93b214b6e6e05add8fc4940df95c3aa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_150ede426e9e7ee129b7205340691eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[8816], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c420767e2c0aee96602b571bfe8f0e3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[247], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_797af715e3943298ba8996c984a17faa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[3800], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_93b214b6e6e05add8fc4940df95c3aa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[950], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_be6c87ea3905a3cb742a939a31e9c133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bb19115e604275b814c3e9625c3fde01
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
                paddle.cast(paddle.randint(low=0, high=2, shape=[70], dtype='int32'), 'bool'),
            ]


    

if __name__ == '__main__':
    unittest.main()