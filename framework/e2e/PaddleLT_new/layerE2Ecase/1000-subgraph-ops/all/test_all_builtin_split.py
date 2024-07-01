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
    class PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebd9ba319153fd3aa806ea44881d17b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a7c504d67503dca941b4c6b128c758d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94b802c1d1caa949fb2c43dbe28381bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49fc4b59da4fe013db237f7ba4836aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    
    class PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdafbdd096b3d7b9e8fea7a9ca85c62c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([300], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_f9028c4517a7e61287ccbbac2353003f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3f8585c63aabf8328ad1906f75e2112(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1ba497b782b1870e6b887b5426623bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_d2b92d8fd6e65b53fbda81105dd128bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a33f9b1b9b2fa73ab67c11d378b69026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13313806056976318, 0.4626670777797699, 0.46212807297706604, 0.25356119871139526], [0.12604497373104095, 0.47608232498168945, 0.47727617621421814, 0.3902834355831146]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_574a1c4e660930d097289115d9e54998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_d4eb8db860772284c18f73070c027a02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54450736a08440987188003139be5f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e84becd0564ec79abe30116b2d211bcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3b6cd8129dcf1c5d33455a4345c9b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_239184908e1d5a33380973011cd34f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([100], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9fabd72ab60334d830aa23d46f2eb7a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ca75ebab4b49fc0d2cdce143fc2e371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bedcfa00c31b60a590bdb617e07b9fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de2198d59b99d80b1f6bd759095e8776(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_015e287d23739d8848beee84071b7a00(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1be5a04d817cd5d6b2b9de980de3060d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_015e287d23739d8848beee84071b7a00
        def get_inputs(self):
            return [
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_67ce7ea0dfb64699ad5e9d7274b08429(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba93b437fa58e880537d7390a0769903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67ce7ea0dfb64699ad5e9d7274b08429
        def get_inputs(self):
            return [
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8363f57410cf7b5cba9746b89fd607dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87789de64b7091492c892e2b6a84aa8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8363f57410cf7b5cba9746b89fd607dc
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1a49c680e0f3264e449fd85b7bae6e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1be5a04d817cd5d6b2b9de980de3060d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_015e287d23739d8848beee84071b7a00
        def get_inputs(self):
            return [
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba93b437fa58e880537d7390a0769903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67ce7ea0dfb64699ad5e9d7274b08429
        def get_inputs(self):
            return [
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87789de64b7091492c892e2b6a84aa8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8363f57410cf7b5cba9746b89fd607dc
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b617d8e8d2b86579fa86e2f2a2ac6cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23f12d8986fe66192d2b814eaac6e0c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be90340626b01f50a00df570d1f7e067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a5367fd2be4937917e77e2660f67103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08681345731019974, 0.18166474997997284, 0.365119993686676, 0.0067316764034330845], [0.10514085739850998, 0.4515710771083832, 0.0026658447459340096, 0.3969408869743347]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_574a1c4e660930d097289115d9e54998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_cd1f71e8fcb2900626d0b9bcc4bfd878(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a87bb8e4e4bcd83929a833a0e80cac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd1f71e8fcb2900626d0b9bcc4bfd878
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e06e89d1ef8ed022ca02fc1fc0ba075d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_978304c3ac13fc14297660599ed70a74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0e550c7f7fcb3c155fcac7eb385939c5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_263524c27c572d2b91c56aceb6d24961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e550c7f7fcb3c155fcac7eb385939c5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.09539089351892471, 0.014189804904162884]], [[0.4132586717605591, 0.26804450154304504]], [[0.2966609001159668, 0.13498319685459137]], [[0.1327727735042572, 0.43529269099235535]], [[0.22661817073822021, 0.15752831101417542]], [[0.050448279827833176, 0.11166179925203323]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.22274713218212128, 0.11812928318977356]], [[0.40234601497650146, 0.32135093212127686]], [[0.24842727184295654, 0.3543384373188019]], [[0.4109976589679718, 0.028806988149881363]], [[0.09972511976957321, 0.48537272214889526]], [[0.24782226979732513, 0.12431900948286057]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.3776187300682068, 0.03984673321247101]], [[0.05422133579850197, 0.33713653683662415]], [[0.0584227554500103, 0.3913451135158539]], [[0.33883368968963623, 0.37223076820373535]], [[0.41833019256591797, 0.4377310276031494]], [[0.4809417724609375, 0.16457433998584747]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.025420304387807846, 0.09367721527814865]], [[0.14023251831531525, 0.34991103410720825]], [[0.418991357088089, 0.374828040599823]], [[0.07487571239471436, 0.2823382019996643]], [[0.3401561975479126, 0.3036860525608063]], [[0.23089678585529327, 0.15044544637203217]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    
    class PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_335c2b79219b83ea11b6fd7b4ab0899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2746506929397583, 0.3692845404148102], [0.2529679834842682, 0.3538026511669159], [0.1918400079011917, 0.46419987082481384], [0.4446069300174713, 0.48151710629463196], [0.22252815961837769, 0.22237402200698853], [0.07176125049591064, 0.38622698187828064]]], dtype='float32').reshape([1, 6, 2]),
                paddle.to_tensor([[[0.35885387659072876, 0.02352176047861576], [0.1643000692129135, 0.19557800889015198], [0.1283266544342041, 0.21114155650138855], [0.1822909265756607, 0.2833311855792999], [0.12502682209014893, 0.40927043557167053], [0.10549148172140121, 0.02151423506438732]]], dtype='float32').reshape([1, 6, 2]),
                paddle.to_tensor([[[0.060231756418943405], [0.44981682300567627], [0.17508365213871002], [0.4978437125682831], [0.00514345383271575], [0.26088249683380127]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_e7fa67d4bf3242bbe5984c407b32387f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8095d7169a02db0f7ab180c28ef4fd01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e074750f3f63f75f893d975fbe8ea436(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e3190b740564f0491f90602c08aacd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.47624194622039795, 0.4252462089061737, 0.4653246998786926, 0.4718214273452759], [0.26620176434516907, 0.39858391880989075, 0.11420700699090958, 0.23182038962841034]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_574a1c4e660930d097289115d9e54998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_474364cb64ab9acecfb633d659836ab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1ba497b782b1870e6b887b5426623bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_d2b92d8fd6e65b53fbda81105dd128bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_46ba4a102937aa759fc6bf965d983f8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cbb2dbfaa500fc1b60dfdffbe3436dcb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300], dtype='float32'),
                paddle.static.InputSpec(shape=[300], dtype='float32'),
                paddle.static.InputSpec(shape=[300], dtype='float32'),
                paddle.static.InputSpec(shape=[300], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_373f50815e75666d73fedd4dafe4bad8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbb2dbfaa500fc1b60dfdffbe3436dcb
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74eec65aa9288ec9a9dcf394d5146759(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45755ab8cca0770593e7ff6988354b84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74eec65aa9288ec9a9dcf394d5146759
        def get_inputs(self):
            return [
                paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be31fdabbbe138b4db1f742f529d77cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a55627bbfef1ca8edfaa4bfab9d9c7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d704e470eafa7547f84e4eb38ac1961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7fa67d4bf3242bbe5984c407b32387f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76d91a3264432895eb1e31627e17c19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92d8754be7af1860e77364e795ce52fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d6b07ff9354d2ad9dca59a76c872aec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23f12d8986fe66192d2b814eaac6e0c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8dcf3ef510fe00e8d1a8336d16704759(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8dcf3ef510fe00e8d1a8336d16704759(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f99098d6da5c20e1d39f3793b652123(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_880d7a43469f48e71cf262597efe60d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bf3a6399b0d867dc540033188f1a781(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb92df5c617c8a327269bf2e77fb2a84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3123f9251f27f37562ad11f85f25d93(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d12ecc308efead4bcb2ba458da97c4f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3123f9251f27f37562ad11f85f25d93
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_06c341d58b3ebb7519fdada95439138f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98e3ca33bcd659ea3cc54e6bc5a93d15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06c341d58b3ebb7519fdada95439138f
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_898d322b0c7201e2c0bf00205b495a81(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_741a072e635c07e283d0b6828e9b74ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_898d322b0c7201e2c0bf00205b495a81
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cfcb8d7e147dfcafe10fbe25472d714(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d12ecc308efead4bcb2ba458da97c4f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3123f9251f27f37562ad11f85f25d93
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98e3ca33bcd659ea3cc54e6bc5a93d15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06c341d58b3ebb7519fdada95439138f
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_741a072e635c07e283d0b6828e9b74ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_898d322b0c7201e2c0bf00205b495a81
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a24293446d95aebcdc60584a4593c444(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49739229679107666, 0.3351207375526428, 0.4484364688396454, 0.12505020201206207], [0.030114924535155296, 0.19822652637958527, 0.2196095734834671, 0.40188801288604736], [0.37833085656166077, 0.005445803515613079, 0.4010592997074127, 0.42676693201065063], [0.35176393389701843, 0.16488398611545563, 0.10220362991094589, 0.2679820656776428], [0.20115259289741516, 0.23507601022720337, 0.2623601257801056, 0.21225251257419586], [0.3057858347892761, 0.18095499277114868, 0.42079871892929077, 0.12342901527881622], [0.1707804948091507, 0.18216678500175476, 0.05291115120053291, 0.3348720669746399]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b6e93ca8a0910021b221ddc918db5f31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f574b8cce66a157334e8930b3db99920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54450736a08440987188003139be5f2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0d3306cb0099b05a8d3569215188c97c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d37d03834a64e8e6b1a0611c46cb4c4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d5714c09dc0032809f740082851f8d16(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_501405c90ab907de82d3c8d31059f3a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14307871460914612], [0.19619788229465485], [0.1863391399383545], [0.2124066799879074], [0.035387687385082245], [0.1282057762145996], [0.4028562605381012], [0.007067443337291479], [0.31240081787109375]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14178644120693207], [0.06213818117976189], [0.3997913599014282], [0.11639508605003357], [0.015070164576172829], [0.12873531877994537], [0.05166096240282059], [0.269934743642807], [0.09819310158491135]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.13368675112724304], [0.39139324426651], [0.3978815972805023], [0.2982294261455536], [0.2754661738872528], [0.3869280219078064], [0.10337948054075241], [0.465537965297699], [0.4861213266849518]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.16374197602272034], [0.22206810116767883], [0.18589580059051514], [0.26141688227653503], [0.006397350691258907], [0.38319242000579834], [0.3071695864200592], [0.40571993589401245], [0.43421873450279236]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_f22d06054eee6145308b137e245bbf63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.38795778155326843], [0.4817730784416199], [0.017573527991771698], [0.44741085171699524], [0.29538026452064514], [0.3727307915687561], [0.023896757513284683], [0.3053695261478424], [0.20216022431850433]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.49473410844802856], [0.3323400914669037], [0.2790012061595917], [0.4925849139690399], [0.17630773782730103], [0.10014522075653076], [0.08800354599952698], [0.43823152780532837], [0.4259171187877655]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2243945300579071], [0.3421638607978821], [0.4742037057876587], [0.15133216977119446], [0.22187696397304535], [0.2794041037559509], [0.01730276271700859], [0.08600067347288132], [0.0719427838921547]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4538259506225586], [0.18657687306404114], [0.13615421950817108], [0.37509968876838684], [0.4772617220878601], [0.39100539684295654], [0.4677458703517914], [0.004688146524131298], [0.23858243227005005]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_cf4fc0f6dea69e1112df8c4b0086aa5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d846cd6880db3a0a8327ac5d139bdd2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5d1f924341ce44cc8831e813b9ca158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9105df290691f505ede598d7559617f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_961e2289a0997318a8992e4cf678e027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67fd26cb1d4461e85c15446d120332d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 4, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 4, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0140c8f9cad6dc5abe951a7a4e589eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0140c8f9cad6dc5abe951a7a4e589eb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732976eb00c4f9d7b88746c0b1e6b811(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af25528ba5411818850b5569f9189cfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fa4b1319343932b6c5cec580b61fba4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.12931282818317413, 0.48861533403396606, 0.17718838155269623, 0.12520994246006012], [0.07523564994335175, 0.4386076331138611, 0.3620389699935913, 0.07658398151397705], [0.14756430685520172, 0.07572465389966965, 0.26838070154190063, 0.37414035201072693], [0.3915621042251587, 0.19104118645191193, 0.01597001403570175, 0.05103715509176254], [0.22588086128234863, 0.0050764307379722595, 0.31528225541114807, 0.298330157995224], [0.177719846367836, 0.25171852111816406, 0.10758128017187119, 0.3024943470954895]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_fe29c9efde1d0e3e59369ce3065ae09f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([6], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf4fc0f6dea69e1112df8c4b0086aa5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6666e1157321c8db7225a8514fd0a297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0919e16a36092167a6a98a83e9eee367(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 8, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b7a0feecb41c97254a64cfee73729b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0919e16a36092167a6a98a83e9eee367
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_73ceb6fd632a1d4e794f5733437bc7ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_79039f56fd6c84c17640c7728d6b5df8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73ceb6fd632a1d4e794f5733437bc7ce
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63d39ede839c6877cbfdc85f76db7c97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1701001077890396, 0.014843899756669998, 0.09852124750614166, 0.3393995761871338], [0.11748797446489334, 0.2589859068393707, 0.31148210167884827, 0.10683373361825943], [0.3116267919540405, 0.4089503586292267, 0.36851558089256287, 0.13546034693717957]], dtype='float32').reshape([3, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_1e3479ebd47fdbe538ece47b8c37f346(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f7a89b2083844815f017a0b0845cd544(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6c910e495187d526a9caf0ee6ad17fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0ff6ed2f6ff88f18386f3baefb6ca635(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e4b23181e5f3383fe6956df62ff3a821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ff6ed2f6ff88f18386f3baefb6ca635
        def get_inputs(self):
            return [
                paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb21fd904c8a7d778e60ded2f3e494d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c01d7758ca593c3f1f46ed80688b0ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1c0c1836c50a2d46222c36b7447e2d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_959bdd5edbec5dca16a1a6a31c702916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56455ca81b341f4c79c02b6ad7a2e12c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ff6ed2f6ff88f18386f3baefb6ca635
        def get_inputs(self):
            return [
                paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd9906d97d779b3a8a19c0f8d5f1e03b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d84833534a73f27ce1e42e5ac7d7a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_484852396f1ecbca2874c0bcdd76f00b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10de3dc01644337419054491bbde7f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_484852396f1ecbca2874c0bcdd76f00b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45227816700935364, 0.2502661347389221, 0.16833464801311493, 0.15319602191448212, 0.18911296129226685, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.27197179198265076, 0.4429328143596649, 0.17026685178279877, 0.08001313358545303, 0.23915867507457733, 0.05345135182142258], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_9f52e17002596081dfb9b0e36a57ce5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_484852396f1ecbca2874c0bcdd76f00b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.22442752122879028, 0.4289640784263611, 0.301647424697876, 0.0385211743414402], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3058600425720215, 0.14679817855358124, 0.3744381070137024, 0.0002531889476813376, 0.2294905185699463, 0.37590494751930237], dtype='float32').reshape([6]),
                paddle.to_tensor([0.22168461978435516, 0.48501694202423096, 0.36503466963768005, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3964419662952423, 0.30897533893585205, 0.40568625926971436, 0.10646888613700867, 0.4039801359176636, 0.40599721670150757], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0ac4a1a36d7d86d9aee2c1523bc6a90e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ac4a1a36d7d86d9aee2c1523bc6a90e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f99098d6da5c20e1d39f3793b652123(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_467cade233ac99c0d5131c8e27ef8473(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 5, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6e51f9f17d2140928f1b35106507a7be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c93a9a4b241c9516cb5a98c34d92f000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e51f9f17d2140928f1b35106507a7be
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23f12d8986fe66192d2b814eaac6e0c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d846cd6880db3a0a8327ac5d139bdd2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5d1f924341ce44cc8831e813b9ca158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9105df290691f505ede598d7559617f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_474364cb64ab9acecfb633d659836ab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_847947cae58e9d41d3e400519011f30b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.049569591879844666, 0.30775371193885803, 0.3094034790992737, 0.13927102088928223], [0.24713994562625885, 0.37523457407951355, 0.024224577471613884, 0.3512221872806549]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_574a1c4e660930d097289115d9e54998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_8b88496fa4bc5628ccdef40f766c6ff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8f1efabed9dea2dc32b72b9d1632de6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03567306697368622, 0.41631054878234863, 0.14406552910804749, 0.48773443698883057]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5d2be45696fbf07d17f9bd2ed996dd9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abc063bf44f95d2e9551bf2273ed7a29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_65ecb05fb834970184f282cdd2aa3072(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_049960647ca4452c6414ce77d8f3e98b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65ecb05fb834970184f282cdd2aa3072
        def get_inputs(self):
            return [
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_129d0b51f70f60a9d7bd1df7d75e29a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e135ae91c481dd4ea1fdfc3d85274ff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_129d0b51f70f60a9d7bd1df7d75e29a7
        def get_inputs(self):
            return [
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0432dd87acc4e9cb45da08051cb7a83a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d6a89114fa28ea0c71f5cb465089085(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0432dd87acc4e9cb45da08051cb7a83a
        def get_inputs(self):
            return [
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d85964667a8d1e9e5e605d53f63c4ca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_049960647ca4452c6414ce77d8f3e98b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65ecb05fb834970184f282cdd2aa3072
        def get_inputs(self):
            return [
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e135ae91c481dd4ea1fdfc3d85274ff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_129d0b51f70f60a9d7bd1df7d75e29a7
        def get_inputs(self):
            return [
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d6a89114fa28ea0c71f5cb465089085(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0432dd87acc4e9cb45da08051cb7a83a
        def get_inputs(self):
            return [
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_70f1b6f9bbd8b87ce865f3e8e034108d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 49, 8, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 49, 8, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7aaf28413f05d86c8d43b46ae1463d22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70f1b6f9bbd8b87ce865f3e8e034108d
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea94404777c9d1830fac7d49bd362c92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c8e19745bad1416ee92e431e42bc192b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b51289903ed21e35e56c0db3c920cbab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8e19745bad1416ee92e431e42bc192b
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70561e07e7778124f123547147105388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7e883cf32eda8400f7b754e890e485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_849e1ef0b3626ba258da9d130e44c548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07e092ef708715b6a2a2edb246f21f99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6e93ca8a0910021b221ddc918db5f31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3ab7770b07bc93a633dad513a2a5363(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c351b9e1b99895a7f6b0b01130c5ccb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c351b9e1b99895a7f6b0b01130c5ccb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0507d09efa80fc597c786f1140a7f730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_547b6212391defcf32618a8bac6f5ae4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d810595a93e8dedb42e9bde606ed35d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82d6940e11cd659d26818c2da5a53293(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33109039068222046, 0.08781209588050842, 0.023874972015619278, 0.4068280756473541], [0.27392226457595825, 0.46916234493255615, 0.3789480924606323, 0.32833385467529297], [0.3113239109516144, 0.34468361735343933, 0.4564971327781677, 0.373810350894928], [0.20377878844738007, 0.48164132237434387, 0.3123581111431122, 0.20703266561031342], [0.44531503319740295, 0.4649951756000519, 0.18325623869895935, 0.19394664466381073], [0.322318434715271, 0.3634970784187317, 0.12563112378120422, 0.49102601408958435], [0.2974511981010437, 0.2236485332250595, 0.2726476192474365, 0.3250909149646759]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_189a4b182482815084443de7253080d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4272349774837494, 0.47877711057662964, 0.08770851790904999, 0.08134938776493073]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_625cfa2afdb678c3a319982a7a3e014c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eb484b46c4208487966929889a7d2213(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64052920912824595f260f8764e205d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb484b46c4208487966929889a7d2213
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_07e092ef708715b6a2a2edb246f21f99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_427d2cffd634898b9fc045d06ceac239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.178619846701622]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.15484283864498138]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.27537572383880615]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.19937725365161896]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_6f74d7e03a96d3fe4b1fd59a1a055290(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20313051342964172]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.004780620336532593]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.31011876463890076]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.14600974321365356]], dtype='float32').reshape([1, 1]),
            ]


    
    class PrimitiveOp_aac0e886ebeb61f10445478feb85b98f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 196, 8, None], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196, 8, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_542745db891b75ecee857c8b8a7480cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aac0e886ebeb61f10445478feb85b98f
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cabb5a1d2d344c771d653498a8a78c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3637081980705261], [0.39640137553215027], [0.31833428144454956], [0.4412391483783722], [0.11331885308027267], [0.4164228141307831]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.2915404736995697], [0.06591049581766129], [0.19010786712169647], [0.17498598992824554], [0.26003992557525635], [0.24949562549591064]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.28181594610214233], [0.256033331155777], [0.3946605324745178], [0.4794350862503052], [0.0947040542960167], [0.32221317291259766]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4542856514453888], [0.32371410727500916], [0.47006481885910034], [0.24301113188266754], [0.31819093227386475], [0.3337242603302002]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_465aec22ee16420260c4c84354d3fa3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.269661545753479], [0.24576835334300995], [0.09308407455682755], [0.3879702687263489], [0.43500569462776184], [0.14424768090248108]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.25507786870002747], [0.1776416003704071], [0.0561361089348793], [0.35384827852249146], [0.1015116423368454], [0.23805175721645355]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0779203251004219], [0.28568974137306213], [0.2259289026260376], [0.08137910068035126], [0.37925606966018677], [0.45194125175476074]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4113561809062958], [0.38559842109680176], [0.34662380814552307], [0.25977855920791626], [0.09847179055213928], [0.15297983586788177]], dtype='float32').reshape([6, 1]),
            ]


    
    class PrimitiveOp_78bf6f471b2704ad0647c5f0b38760d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 300, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e663623d60a7bef73df1ca5db74e921(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78bf6f471b2704ad0647c5f0b38760d5
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ca75ebab4b49fc0d2cdce143fc2e371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2d782e9f679f61de346ecd3c3ed5799(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_379be5ccbf78cb8f90b2b8fa21701de2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2d782e9f679f61de346ecd3c3ed5799
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d21c57512999f3e701b3c8c43538c8ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2619705c7fe663d42b4682987ae91f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d21c57512999f3e701b3c8c43538c8ad
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8c9e55b393bc6b2080654bce9a7f9e16(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6dbd2a3e7755c0327f61bd058d35ab28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8c9e55b393bc6b2080654bce9a7f9e16
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_096e7b2f9b19a5c26dbc397eeb763a3b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_701dbadea060dcc1f27ab8108e86a9e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_096e7b2f9b19a5c26dbc397eeb763a3b
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54dbdfc5b52bbf8483a0d7ed2cc334db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80931218eb525d64a291884f999810ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2777930796146393, 0.02657657489180565, 0.47445645928382874, 0.20409588515758514], [0.1387108713388443, 0.39539310336112976, 0.1251879781484604, 0.12581704556941986], [0.0013369943480938673, 0.06631842255592346, 0.38734182715415955, 0.07199130952358246], [0.4746476411819458, 0.005258830729871988, 0.14967621862888336, 0.14246855676174164], [0.15963207185268402, 0.3332064747810364, 0.09021896123886108, 0.44095081090927124]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_febbde00d51f0611de77faf3e99724ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([5], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c60317cee82ebe03088e71b719e4e79c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36710721254348755, 0.2733869254589081, 0.12741735577583313, 0.09983520209789276], [0.1689828783273697, 0.12775754928588867, 0.4929034411907196, 0.024316783994436264], [0.28232342004776, 0.3154520094394684, 0.34447357058525085, 0.1899302452802658], [0.3894905745983124, 0.21266524493694305, 0.15552382171154022, 0.2049335241317749], [0.14210514724254608, 0.42684876918792725, 0.16730007529258728, 0.03494313359260559], [0.36946699023246765, 0.25839653611183167, 0.40660321712493896, 0.049302250146865845], [0.4219869375228882, 0.046291451901197433, 0.45390188694000244, 0.2299237996339798]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_07e092ef708715b6a2a2edb246f21f99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a592a3fbb3c9fdeb2ce8b5859624cd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04564209654927254, 0.4040566384792328, 0.4310336410999298, 0.19761954247951508], [0.06333372741937637, 0.015020381659269333, 0.26050060987472534, 0.4925742447376251], [0.004450720734894276, 0.09676632285118103, 0.15310807526111603, 0.2296193391084671], [0.3995705544948578, 0.20686708390712738, 0.16882404685020447, 0.41516441106796265], [0.10278435796499252, 0.2873406708240509, 0.33401980996131897, 0.45226940512657166], [0.16829141974449158, 0.3355092406272888, 0.43085777759552, 0.3128385841846466], [0.39289021492004395, 0.4308372735977173, 0.11628603935241699, 0.23280969262123108]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cab61b152748f4449b087b0315445390(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbd365ede173252cda7fb38ca83e54d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_072b245602a46c4f73b07f6852f6ff70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06026938185095787, 0.10351205617189407, 0.010360955260694027, 0.05767056718468666]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_49fc4b59da4fe013db237f7ba4836aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_cdafbdd096b3d7b9e8fea7a9ca85c62c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([300], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_9890c1f003db87470310356f77800c89(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b0e3d46d8c13b538b0d4491938e4e171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9890c1f003db87470310356f77800c89
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96eaa311925eb222c0db43395143a0a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e853096623867803a90faa18bc8cbf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e853096623867803a90faa18bc8cbf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59fae8a71cb139482d2ac9589425a053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d7510ccc96730a9120be9b9c2b277b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 90, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33d17eaa239d9b6650f97f21d562ca4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d7510ccc96730a9120be9b9c2b277b4
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87789de64b7091492c892e2b6a84aa8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8363f57410cf7b5cba9746b89fd607dc
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f454df70ac5248e2da974cdc186ea04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9423ab8ac566408ad429a45a5c39888(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3ac8de11e24cb589258612ad7601f877(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_782ad5e19dba50223bbe8f39783fc92e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ac8de11e24cb589258612ad7601f877
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8095d7169a02db0f7ab180c28ef4fd01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4661e8efa0048a7d863480abde00efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4661e8efa0048a7d863480abde00efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62ff55f05ef3c28eccc6064cabe177b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3befe3cc79dd7f4776586e7595d7ff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d3348bb2822b4aa8623fe0975ad1e5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 600, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 600, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ffd0a5228c73cb23b312c86604e20486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d3348bb2822b4aa8623fe0975ad1e5c
        def get_inputs(self):
            return [
                paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fd790ae02adc0ab1d16032201e49068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.11586524546146393, 0.3802587389945984, 0.2655188739299774, 0.32558372616767883], [0.2183745801448822, 0.062354110181331635, 0.09305060654878616, 0.19626672565937042], [0.051588743925094604, 0.2933014929294586, 0.06933026015758514, 0.34952929615974426], [0.4719468057155609, 0.09559851884841919, 0.3501664698123932, 0.04593765735626221], [0.3326735496520996, 0.0424429252743721, 0.33680230379104614, 0.1775297373533249]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_febbde00d51f0611de77faf3e99724ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([5], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f574b8cce66a157334e8930b3db99920(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e044e2dd467c4f5794474fae2591e948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e044e2dd467c4f5794474fae2591e948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdf351ca22a016a3bdd8985ba0f76d0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_474364cb64ab9acecfb633d659836ab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adc64b8b9edc4141f2fcb4cdf33a09b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b8bb476bd1f2e4205f663af8c319bc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba7be401d6ba5c7431f7b0c9a2fc868d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08afa045dacac46f4c742d8024111a7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d1b95a18e1e6684b3fd855eae577955(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e46fa36943e61e60a370e8494835d0b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6242d6371f5ecb876463e616c69d5e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e51f9f17d2140928f1b35106507a7be
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e5c0e379201d3c08c4bc27cf20b30900(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e64d1b9a9af596bf0bb3ef0a30e7c826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5c0e379201d3c08c4bc27cf20b30900
        def get_inputs(self):
            return [
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d48936e941bd51091e548d7aa5f8f950(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebb3ff3c8befc52a5e30b773902c9064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d48936e941bd51091e548d7aa5f8f950
        def get_inputs(self):
            return [
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cde7d3528ca538fb980be1a81e8c768c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c13eff32d8254befd61873272eab10d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cde7d3528ca538fb980be1a81e8c768c
        def get_inputs(self):
            return [
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e572f7dc80b19dab69cd9ca1965de660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e64d1b9a9af596bf0bb3ef0a30e7c826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5c0e379201d3c08c4bc27cf20b30900
        def get_inputs(self):
            return [
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebb3ff3c8befc52a5e30b773902c9064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d48936e941bd51091e548d7aa5f8f950
        def get_inputs(self):
            return [
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c13eff32d8254befd61873272eab10d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cde7d3528ca538fb980be1a81e8c768c
        def get_inputs(self):
            return [
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fb2946e234bfa619c2e298006e391f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25348156690597534], [0.1419389396905899], [0.21522848308086395], [0.18140600621700287], [0.35213175415992737]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.47471386194229126], [0.33389967679977417], [0.40402311086654663], [0.4485560953617096], [0.46584224700927734]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.18775731325149536], [0.12440189719200134], [0.05388212949037552], [0.37235766649246216], [0.08130541443824768]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21230792999267578], [0.0743480697274208], [0.0162972342222929], [0.3133012354373932], [0.06640253961086273]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_25faea9d13af884adc9dcd4580c10dfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1649070680141449], [0.39336472749710083], [0.47650161385536194], [0.22819465398788452], [0.16766460239887238]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21730045974254608], [0.35416045784950256], [0.32481035590171814], [0.011503858491778374], [0.4248878061771393]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.13528534770011902], [0.3804410994052887], [0.13431361317634583], [0.32307666540145874], [0.13232417404651642]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4135350286960602], [0.022249840199947357], [0.3079543709754944], [0.28616321086883545], [0.11313261836767197]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_e7152b4b774025650f3d862d91725cb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3226352334022522, 0.1387707144021988, 0.13630889356136322, 0.3679457902908325]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_26b66ad791a671c02914f226ed4f5d82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08afa045dacac46f4c742d8024111a7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c320601e476299e5fe459b24918ad56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d29063b61c5fba7680036558a84fd24c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b53834586a8897ff3299f17e268f5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6a7d557a7adf227e179b2dbcdf8ca3ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90f4d2bb62d1036c359f9795a0f194ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6a7d557a7adf227e179b2dbcdf8ca3ac
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7246149cdac320e103819305e35c4ee8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17b679b3a753eb69f81cfec1efdf776e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_050bf06a732284897bf204e008c13689(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b679b3a753eb69f81cfec1efdf776e
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bdb024a8baa0218063cb0424ec2e4db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e16aaaa2745e6ee68df1772397b6b30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4692842364311218, 0.07899530231952667, 0.43798011541366577, 0.0650915876030922]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c747fb5aabfc2d159d84a5bf8277fb67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1917f5ca69fa5739870ab954160a9c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cbbabfd98cbda7c561a4022e56d2387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cbbabfd98cbda7c561a4022e56d2387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7883a2bf3572fd75101117f35c9dfdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7e883cf32eda8400f7b754e890e485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1ba497b782b1870e6b887b5426623bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_d2b92d8fd6e65b53fbda81105dd128bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5ebb54ca0294d839e3aeb80b243ea4d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ebb54ca0294d839e3aeb80b243ea4d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e572f7dc80b19dab69cd9ca1965de660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9408af38c22fc1f6d4528c04939c951(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9408af38c22fc1f6d4528c04939c951(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccfcbbc7044a9666120e0e95456ffeca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_849b26d1b5bc39f2090aefc80555c90c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3b6cd8129dcf1c5d33455a4345c9b50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_239184908e1d5a33380973011cd34f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([100], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_310f18485eda0529841cc4f7684fc7ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_077a2106935067b579cc3453acbed413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7ad2e63a4498a0de722a173df8c8f95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_894874e3b1094cdd9a58f17408c93835(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 60, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_714aedeb92ac6ffff6818934900ccefb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_894874e3b1094cdd9a58f17408c93835
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20475a276f2d4567c38a3392e8ecfe18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17b679b3a753eb69f81cfec1efdf776e
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f34fd3226f19e3e92385608d10b6e71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_74bf011788e17704ea50188e82c018b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f79fdaf27346adbd050b77ecced1920e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74bf011788e17704ea50188e82c018b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d2b41fcd6e4c8d7dd411ba9df71367a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b21903df8754f8b83833789e6fe92732(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2b41fcd6e4c8d7dd411ba9df71367a9
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_43faf20a11784193ef2ef3f214e4a626(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_196512f4a6f1947a56b86ea910da6fe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43faf20a11784193ef2ef3f214e4a626
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_535c2d7496bd9d26d6aa09677dbd457d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9cd4ef745dff3bf9b21c71c8a94a171a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_535c2d7496bd9d26d6aa09677dbd457d
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1f0cd93a22e1906e16fe7f46dcd46913(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e4ef8d17956da98087dbeff7c11a607(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f0cd93a22e1906e16fe7f46dcd46913
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3f6a36d5d855a83444cc70fd95bf0fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce4f12ad7c813baadb837d25725eb8ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3f6a36d5d855a83444cc70fd95bf0fd
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8e6553e07d549e21f2a6b561f0551aa3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[3800, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[950, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[247, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[70, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2a0eff166684bbfc8f9f3ebafea5480(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8e6553e07d549e21f2a6b561f0551aa3
        def get_inputs(self):
            return [
                paddle.uniform([15200, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88de75d7bdedc2ebee0e6c7bec03a411(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5bb2ad36ca7dc8e8b869423f323e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af25528ba5411818850b5569f9189cfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5a901d4baa11e46da4e11953f0be2044(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 20, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7b6b30e194d716085015bbe0cebcca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a901d4baa11e46da4e11953f0be2044
        def get_inputs(self):
            return [
                paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8116b47fbdc824ec9d76847fac1a2e29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e074750f3f63f75f893d975fbe8ea436(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf7e883cf32eda8400f7b754e890e485(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d26ae9a79cbdda86a8a71d962e9d1524(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c2af12218506b683b179a237a89615ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d26ae9a79cbdda86a8a71d962e9d1524
        def get_inputs(self):
            return [
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19f0e73dfcead485525c521ff14866a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9890c1f003db87470310356f77800c89
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_76295f48ff6a8b66fe06d5fb1db93144(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 180, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6f68e11ceda3b93cffebf93b419a833(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_76295f48ff6a8b66fe06d5fb1db93144
        def get_inputs(self):
            return [
                paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a096e22436d7e9adb56a573fa5918772(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2400294840335846, 0.33962640166282654, 0.2767137289047241, 0.09159930795431137]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1e8720eba2ee7060749ba56bcf2cbd7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.43446049094200134], [0.3114164471626282], [0.21863189339637756], [0.22740739583969116]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28952881693840027], [0.0999356284737587], [0.3750646412372589], [0.1684504747390747]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.1857021450996399], [0.232812762260437], [0.44483304023742676], [0.43192264437675476]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28646060824394226], [0.2347613275051117], [0.07614689320325851], [0.21912746131420135]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_c9e5d00474362dcf69f4216398e0c4f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05875740945339203], [0.3892061412334442], [0.20329052209854126], [0.3478715717792511]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2514371871948242], [0.428699254989624], [0.0340176485478878], [0.16604164242744446]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.37367141246795654], [0.26348745822906494], [0.22833964228630066], [0.3062015175819397]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3861541450023651], [0.4490680992603302], [0.4151049554347992], [0.03166484087705612]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_a47c468ae4db47a8f6efcbb5ea37c944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af25528ba5411818850b5569f9189cfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5028ef7e1ce7dfa7a6d009c77f9e860f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_096e7b2f9b19a5c26dbc397eeb763a3b
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f34fd3226f19e3e92385608d10b6e71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19c04f3bc9f18c31822139a67b33004d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19c04f3bc9f18c31822139a67b33004d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59fae8a71cb139482d2ac9589425a053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e06e89d1ef8ed022ca02fc1fc0ba075d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7db1793e30438f3cf0a01fb3997fc476(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4eb8db860772284c18f73070c027a02
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d285961161ffbd4571cc6515571941e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196, 4, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196, 4, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_49b88467791285d01cb9d678bc0112aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d285961161ffbd4571cc6515571941e9
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e4f82265497225fdd3309b546738d6cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5187d777b7b9055bf4a4bddac4de0a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4f82265497225fdd3309b546738d6cb
        def get_inputs(self):
            return [
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0703cb29f417261f874adef0da106ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c371e77e8dec85a7b226b94b3fdda33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0703cb29f417261f874adef0da106ea
        def get_inputs(self):
            return [
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0d44b67909e72f07522ee9c2491ad097(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
                paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ad57bfc286453aa96391be6d47ade35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d44b67909e72f07522ee9c2491ad097
        def get_inputs(self):
            return [
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec15443986f3f393b7b731906147d28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5187d777b7b9055bf4a4bddac4de0a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4f82265497225fdd3309b546738d6cb
        def get_inputs(self):
            return [
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c371e77e8dec85a7b226b94b3fdda33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0703cb29f417261f874adef0da106ea
        def get_inputs(self):
            return [
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ad57bfc286453aa96391be6d47ade35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d44b67909e72f07522ee9c2491ad097
        def get_inputs(self):
            return [
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6fee5e82968273fc3911d84974f0fe69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4faf06fc9799a11ee21974abd327df3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fee5e82968273fc3911d84974f0fe69
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9ae3e2555380f8e2c71740d43d7e28ca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 16, 12, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 16, 12, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb1beba7b5f13a8d5b6455bed18274ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ae3e2555380f8e2c71740d43d7e28ca
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60802e34ec64ac3dcd57495a62ea21c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d0d9ed7e16c61642bd9c06c26924fc6
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e69887eefd29250dd6b7644b5ff9ae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9028c4517a7e61287ccbbac2353003f
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ac295eabc4d4dbe2437f79f9be0fe09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be86e4a33fef4950a90444d205801ccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be86e4a33fef4950a90444d205801ccf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a74f60bd28fa6aaa57257cb1399a48e1
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d85964667a8d1e9e5e605d53f63c4ca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_654b90c99a208e056b651f331b21b979(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13706611096858978, 0.08502600342035294, 0.44341975450515747, 0.3082224726676941], [0.009993945248425007, 0.3372018337249756, 0.11831139028072357, 0.047583818435668945], [0.4995841383934021, 0.01382436789572239, 0.0017661130987107754, 0.08258343487977982], [0.42873167991638184, 0.27258726954460144, 0.20817619562149048, 0.4982690215110779], [0.24455808103084564, 0.42871490120887756, 0.45946192741394043, 0.36610716581344604], [0.05611858889460564, 0.20213836431503296, 0.16467683017253876, 0.09433294087648392], [0.07387426495552063, 0.3788360059261322, 0.16099436581134796, 0.3784846365451813]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b471826d528975b71c6ce493dcc1c736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3ad5325883aa81debc0b8dca6843d2a3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29233288764953613, 0.36788859963417053, 0.006307649426162243, 0.2807435095310211], [0.3606562316417694, 0.492276668548584, 0.20333808660507202, 0.06060683727264404], [0.14438258111476898, 0.47764408588409424, 0.38392844796180725, 0.3385300934314728], [0.037337061017751694, 0.3944132924079895, 0.4904634356498718, 0.26613932847976685], [0.280810683965683, 0.27498728036880493, 0.3712540864944458, 0.056545354425907135], [0.42267999053001404, 0.3459642827510834, 0.30062058568000793, 0.436982125043869]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_fe29c9efde1d0e3e59369ce3065ae09f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([6], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_e72b9b55629a7db8024e76f14d0d7516(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74bf011788e17704ea50188e82c018b6
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b476bd9383df97da1214d00f77356eaf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 49, 16, None], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 49, 16, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_982dae35a1dd5964d0dec97d718f4164(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b476bd9383df97da1214d00f77356eaf
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebd9ba319153fd3aa806ea44881d17b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 21504, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94b802c1d1caa949fb2c43dbe28381bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80cb3b8bc81d2ab5df2b5d2fd5811f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_cdafbdd096b3d7b9e8fea7a9ca85c62c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([300], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_d5946d096887df11f977a77c30054289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 24, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6832866c56cbca1ada98e2d14c7f6f3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_d2b92d8fd6e65b53fbda81105dd128bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_fb83d99d07cd67885f72bbcbf8585b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13313806056976318, 0.4626670777797699, 0.46212807297706604, 0.25356119871139526], [0.12604497373104095, 0.47608232498168945, 0.47727617621421814, 0.3902834355831146]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_574a1c4e660930d097289115d9e54998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7f99098d6da5c20e1d39f3793b652123(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e84becd0564ec79abe30116b2d211bcf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a85593988e43a6a60cee31d5b9e17811(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_239184908e1d5a33380973011cd34f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([100], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_27b28cb0df3d7ea8b77a12cc863da226(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 112, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad1f692859f02aab7389bb31deced781(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d4dbb330f83a18463bbd165bd2a3c16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de2198d59b99d80b1f6bd759095e8776(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68babfcbf9e78282bbf6db058fd40d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97e93d372fa6f874faab8c4704396502(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e99bb3ecf818b292ae55f6e1298208e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1a49c680e0f3264e449fd85b7bae6e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68babfcbf9e78282bbf6db058fd40d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97e93d372fa6f874faab8c4704396502(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e99bb3ecf818b292ae55f6e1298208e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c83b117fc8a92c6d02f7370b7e8c0fcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_393ed428caf4f0b3bb1364b310dabf09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_277467eaef50fa1ab68d10da7030ba77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 11, 11], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_888e7203795d8130b4baeb8c75b18a78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.08681345731019974, 0.18166474997997284, 0.365119993686676, 0.0067316764034330845], [0.10514085739850998, 0.4515710771083832, 0.0026658447459340096, 0.3969408869743347]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_574a1c4e660930d097289115d9e54998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2add71fe7416737b50e0f9c15b6977a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b27e976c2bd75a9c04062d9bbe6ba8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2add71fe7416737b50e0f9c15b6977a7
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_145482f3a29899a037f6c4d66dac8c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13ab96788a24fed425df8147ad3c5c46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 9, 9], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_263524c27c572d2b91c56aceb6d24961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e550c7f7fcb3c155fcac7eb385939c5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.09539089351892471, 0.014189804904162884]], [[0.4132586717605591, 0.26804450154304504]], [[0.2966609001159668, 0.13498319685459137]], [[0.1327727735042572, 0.43529269099235535]], [[0.22661817073822021, 0.15752831101417542]], [[0.050448279827833176, 0.11166179925203323]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.22274713218212128, 0.11812928318977356]], [[0.40234601497650146, 0.32135093212127686]], [[0.24842727184295654, 0.3543384373188019]], [[0.4109976589679718, 0.028806988149881363]], [[0.09972511976957321, 0.48537272214889526]], [[0.24782226979732513, 0.12431900948286057]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.3776187300682068, 0.03984673321247101]], [[0.05422133579850197, 0.33713653683662415]], [[0.0584227554500103, 0.3913451135158539]], [[0.33883368968963623, 0.37223076820373535]], [[0.41833019256591797, 0.4377310276031494]], [[0.4809417724609375, 0.16457433998584747]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([[[[0.025420304387807846, 0.09367721527814865]], [[0.14023251831531525, 0.34991103410720825]], [[0.418991357088089, 0.374828040599823]], [[0.07487571239471436, 0.2823382019996643]], [[0.3401561975479126, 0.3036860525608063]], [[0.23089678585529327, 0.15044544637203217]]]], dtype='float32').reshape([1, 6, 1, 2]),
            ]


    class TestPrimitiveOp_335c2b79219b83ea11b6fd7b4ab0899b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2746506929397583, 0.3692845404148102], [0.2529679834842682, 0.3538026511669159], [0.1918400079011917, 0.46419987082481384], [0.4446069300174713, 0.48151710629463196], [0.22252815961837769, 0.22237402200698853], [0.07176125049591064, 0.38622698187828064]]], dtype='float32').reshape([1, 6, 2]),
                paddle.to_tensor([[[0.35885387659072876, 0.02352176047861576], [0.1643000692129135, 0.19557800889015198], [0.1283266544342041, 0.21114155650138855], [0.1822909265756607, 0.2833311855792999], [0.12502682209014893, 0.40927043557167053], [0.10549148172140121, 0.02151423506438732]]], dtype='float32').reshape([1, 6, 2]),
                paddle.to_tensor([[[0.060231756418943405], [0.44981682300567627], [0.17508365213871002], [0.4978437125682831], [0.00514345383271575], [0.26088249683380127]]], dtype='float32').reshape([1, 6, 1]),
            ]


    class TestPrimitiveOp_dc1d088f8c0fe1d4f727d3b947a82713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7180bf41c7a391ed8e176d6a70eea5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d813eff4107fbd7e5ae7b21ba3cae4bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76b28e0b7a528383c5312073e8e64dc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.47624194622039795, 0.4252462089061737, 0.4653246998786926, 0.4718214273452759], [0.26620176434516907, 0.39858391880989075, 0.11420700699090958, 0.23182038962841034]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_574a1c4e660930d097289115d9e54998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4e730df7b9c3fdb6eecdc66c804f4c15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6832866c56cbca1ada98e2d14c7f6f3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_d2b92d8fd6e65b53fbda81105dd128bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ccfcbbc7044a9666120e0e95456ffeca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e28005b051ab351162963db51ee2bcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_484852396f1ecbca2874c0bcdd76f00b
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd9e3582258a431d8ffe1f5522e58216(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 40, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be31fdabbbe138b4db1f742f529d77cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a55627bbfef1ca8edfaa4bfab9d9c7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7883a2bf3572fd75101117f35c9dfdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc1d088f8c0fe1d4f727d3b947a82713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d68e405033082946f3f1d486b93551(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92d8754be7af1860e77364e795ce52fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d6b07ff9354d2ad9dca59a76c872aec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_393ed428caf4f0b3bb1364b310dabf09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1db0226e926d3ad390529b9d538fb654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1db0226e926d3ad390529b9d538fb654(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f99098d6da5c20e1d39f3793b652123(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_880d7a43469f48e71cf262597efe60d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d85964667a8d1e9e5e605d53f63c4ca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb92df5c617c8a327269bf2e77fb2a84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c320601e476299e5fe459b24918ad56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cfcb8d7e147dfcafe10fbe25472d714(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c320601e476299e5fe459b24918ad56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a53907179c731ad9709960ba9e007946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.49739229679107666, 0.3351207375526428, 0.4484364688396454, 0.12505020201206207], [0.030114924535155296, 0.19822652637958527, 0.2196095734834671, 0.40188801288604736], [0.37833085656166077, 0.005445803515613079, 0.4010592997074127, 0.42676693201065063], [0.35176393389701843, 0.16488398611545563, 0.10220362991094589, 0.2679820656776428], [0.20115259289741516, 0.23507601022720337, 0.2623601257801056, 0.21225251257419586], [0.3057858347892761, 0.18095499277114868, 0.42079871892929077, 0.12342901527881622], [0.1707804948091507, 0.18216678500175476, 0.05291115120053291, 0.3348720669746399]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_a2c54adcdf016b5c317517228fa4fb80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_642319a6cbb99934b4ea44f577a39c83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f99098d6da5c20e1d39f3793b652123(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d37d03834a64e8e6b1a0611c46cb4c4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501405c90ab907de82d3c8d31059f3a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.14307871460914612], [0.19619788229465485], [0.1863391399383545], [0.2124066799879074], [0.035387687385082245], [0.1282057762145996], [0.4028562605381012], [0.007067443337291479], [0.31240081787109375]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.14178644120693207], [0.06213818117976189], [0.3997913599014282], [0.11639508605003357], [0.015070164576172829], [0.12873531877994537], [0.05166096240282059], [0.269934743642807], [0.09819310158491135]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.13368675112724304], [0.39139324426651], [0.3978815972805023], [0.2982294261455536], [0.2754661738872528], [0.3869280219078064], [0.10337948054075241], [0.465537965297699], [0.4861213266849518]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.16374197602272034], [0.22206810116767883], [0.18589580059051514], [0.26141688227653503], [0.006397350691258907], [0.38319242000579834], [0.3071695864200592], [0.40571993589401245], [0.43421873450279236]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_f22d06054eee6145308b137e245bbf63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.38795778155326843], [0.4817730784416199], [0.017573527991771698], [0.44741085171699524], [0.29538026452064514], [0.3727307915687561], [0.023896757513284683], [0.3053695261478424], [0.20216022431850433]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.49473410844802856], [0.3323400914669037], [0.2790012061595917], [0.4925849139690399], [0.17630773782730103], [0.10014522075653076], [0.08800354599952698], [0.43823152780532837], [0.4259171187877655]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.2243945300579071], [0.3421638607978821], [0.4742037057876587], [0.15133216977119446], [0.22187696397304535], [0.2794041037559509], [0.01730276271700859], [0.08600067347288132], [0.0719427838921547]], dtype='float32').reshape([9, 1]),
                paddle.to_tensor([[0.4538259506225586], [0.18657687306404114], [0.13615421950817108], [0.37509968876838684], [0.4772617220878601], [0.39100539684295654], [0.4677458703517914], [0.004688146524131298], [0.23858243227005005]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_f302f001c87cbdc9f9b773d6021f4b70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d846cd6880db3a0a8327ac5d139bdd2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5d1f924341ce44cc8831e813b9ca158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9105df290691f505ede598d7559617f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_961e2289a0997318a8992e4cf678e027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14340950050d1cd86a3d9ba268c76051(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 4, 4], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 4, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6d6ad4cf10389f6c54f36f3c118a44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b6d6ad4cf10389f6c54f36f3c118a44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732976eb00c4f9d7b88746c0b1e6b811(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed207b7d2541d1cb1d6af37dd414cbad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7eadf1fd1201a8416d6e89a03caee42c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.12931282818317413, 0.48861533403396606, 0.17718838155269623, 0.12520994246006012], [0.07523564994335175, 0.4386076331138611, 0.3620389699935913, 0.07658398151397705], [0.14756430685520172, 0.07572465389966965, 0.26838070154190063, 0.37414035201072693], [0.3915621042251587, 0.19104118645191193, 0.01597001403570175, 0.05103715509176254], [0.22588086128234863, 0.0050764307379722595, 0.31528225541114807, 0.298330157995224], [0.177719846367836, 0.25171852111816406, 0.10758128017187119, 0.3024943470954895]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_fe29c9efde1d0e3e59369ce3065ae09f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([6], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f302f001c87cbdc9f9b773d6021f4b70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 60, 100], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d76b913e42f78c80c99350f7c67e0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1cc7a2959f2178db2aa4722beb4393b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 8, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcedb801a850b68e2c227c43d24e4a59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 48, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aac997be07d4d41f00176c8e7e742b93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1701001077890396, 0.014843899756669998, 0.09852124750614166, 0.3393995761871338], [0.11748797446489334, 0.2589859068393707, 0.31148210167884827, 0.10683373361825943], [0.3116267919540405, 0.4089503586292267, 0.36851558089256287, 0.13546034693717957]], dtype='float32').reshape([3, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_1e3479ebd47fdbe538ece47b8c37f346(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f7a89b2083844815f017a0b0845cd544(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65cf5fa524e14527b0986d168c97ee50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04cf36cb506d8b123af17de3e4dc8d09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 120, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f4b343400f4afe4fd6a84fad794aad2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 44, 44], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c01d7758ca593c3f1f46ed80688b0ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 28, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1c0c1836c50a2d46222c36b7447e2d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_959bdd5edbec5dca16a1a6a31c702916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_422ba5083af672982015cea9f6555d6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd9906d97d779b3a8a19c0f8d5f1e03b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 28, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bf77e1a58832de9b0980ce0ccaffdbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 22, 22], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10de3dc01644337419054491bbde7f88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_484852396f1ecbca2874c0bcdd76f00b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348], dtype='float32').reshape([6]),
                paddle.to_tensor([0.45227816700935364, 0.2502661347389221, 0.16833464801311493, 0.15319602191448212, 0.18911296129226685, 0.1342129111289978], dtype='float32').reshape([6]),
                paddle.to_tensor([0.27197179198265076, 0.4429328143596649, 0.17026685178279877, 0.08001313358545303, 0.23915867507457733, 0.05345135182142258], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_9f52e17002596081dfb9b0e36a57ce5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_484852396f1ecbca2874c0bcdd76f00b
        def get_inputs(self):
            return [
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.22442752122879028, 0.4289640784263611, 0.301647424697876, 0.0385211743414402], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3058600425720215, 0.14679817855358124, 0.3744381070137024, 0.0002531889476813376, 0.2294905185699463, 0.37590494751930237], dtype='float32').reshape([6]),
                paddle.to_tensor([0.22168461978435516, 0.48501694202423096, 0.36503466963768005, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146], dtype='float32').reshape([6]),
                paddle.to_tensor([0.3964419662952423, 0.30897533893585205, 0.40568625926971436, 0.10646888613700867, 0.4039801359176636, 0.40599721670150757], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_a5e295c64827ffef834919e12fd40edc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5e295c64827ffef834919e12fd40edc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f99098d6da5c20e1d39f3793b652123(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3549, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5751a5e5756c853ac9b3f07fd0c6d963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 5, 5], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44144899ff81fae8a06627f54c9eb4d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_393ed428caf4f0b3bb1364b310dabf09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d846cd6880db3a0a8327ac5d139bdd2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 20, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5d1f924341ce44cc8831e813b9ca158(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 40, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9105df290691f505ede598d7559617f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 80, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e730df7b9c3fdb6eecdc66c804f4c15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f1650c3a2c275eabc27a1ab3326b850(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.049569591879844666, 0.30775371193885803, 0.3094034790992737, 0.13927102088928223], [0.24713994562625885, 0.37523457407951355, 0.024224577471613884, 0.3512221872806549]], dtype='float32').reshape([2, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_574a1c4e660930d097289115d9e54998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_59fae8a71cb139482d2ac9589425a053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a63186a8139532e33e5b5be4e9b441a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03567306697368622, 0.41631054878234863, 0.14406552910804749, 0.48773443698883057]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_5d2be45696fbf07d17f9bd2ed996dd9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 17, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b11d7baafcc867f544406c03bbef9f50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c53ef34d48db7aa75861fff390f577bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6126ed602edfcd631f5938f75a823db0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_494a3cdd1edafe72ee9f26f1f3362ae3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d85964667a8d1e9e5e605d53f63c4ca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c53ef34d48db7aa75861fff390f577bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6126ed602edfcd631f5938f75a823db0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_494a3cdd1edafe72ee9f26f1f3362ae3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1231fa6854e4c1b8c174fb529a6d5b14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2add71fe7416737b50e0f9c15b6977a7
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 49, 8, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 49, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3546ac772200ad89041e990399903e32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa1a1202d3e83569584586751cf88931(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2add71fe7416737b50e0f9c15b6977a7
        def get_inputs(self):
            return [
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 160, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40c9885b2305cb42ef915649a3ed855a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5424831fc367eb2d46a01852cc857e88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_849e1ef0b3626ba258da9d130e44c548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e93612014430188ec26ff867f096481(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2c54adcdf016b5c317517228fa4fb80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e572f7dc80b19dab69cd9ca1965de660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5af8d8e40c663b4ec8741f9742aafc75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5af8d8e40c663b4ec8741f9742aafc75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0507d09efa80fc597c786f1140a7f730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7876e5f729c92e008788bdc50bb124a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d810595a93e8dedb42e9bde606ed35d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_441f657fa792f30584613d4edccb0ef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.33109039068222046, 0.08781209588050842, 0.023874972015619278, 0.4068280756473541], [0.27392226457595825, 0.46916234493255615, 0.3789480924606323, 0.32833385467529297], [0.3113239109516144, 0.34468361735343933, 0.4564971327781677, 0.373810350894928], [0.20377878844738007, 0.48164132237434387, 0.3123581111431122, 0.20703266561031342], [0.44531503319740295, 0.4649951756000519, 0.18325623869895935, 0.19394664466381073], [0.322318434715271, 0.3634970784187317, 0.12563112378120422, 0.49102601408958435], [0.2974511981010437, 0.2236485332250595, 0.2726476192474365, 0.3250909149646759]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7c5bca9cbf0569e3f00c4694e7c31218(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4272349774837494, 0.47877711057662964, 0.08770851790904999, 0.08134938776493073]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_625cfa2afdb678c3a319982a7a3e014c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 14, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e53c5dc39b931791fb473ff83e57cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2add71fe7416737b50e0f9c15b6977a7
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e93612014430188ec26ff867f096481(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_427d2cffd634898b9fc045d06ceac239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.178619846701622]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.15484283864498138]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.27537572383880615]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.19937725365161896]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_6f74d7e03a96d3fe4b1fd59a1a055290(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20313051342964172]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.004780620336532593]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.31011876463890076]], dtype='float32').reshape([1, 1]),
                paddle.to_tensor([[0.14600974321365356]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_43958d1d81f1774fc2297fb16977e68f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 8, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cabb5a1d2d344c771d653498a8a78c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3637081980705261], [0.39640137553215027], [0.31833428144454956], [0.4412391483783722], [0.11331885308027267], [0.4164228141307831]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.2915404736995697], [0.06591049581766129], [0.19010786712169647], [0.17498598992824554], [0.26003992557525635], [0.24949562549591064]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.28181594610214233], [0.256033331155777], [0.3946605324745178], [0.4794350862503052], [0.0947040542960167], [0.32221317291259766]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4542856514453888], [0.32371410727500916], [0.47006481885910034], [0.24301113188266754], [0.31819093227386475], [0.3337242603302002]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_465aec22ee16420260c4c84354d3fa3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.269661545753479], [0.24576835334300995], [0.09308407455682755], [0.3879702687263489], [0.43500569462776184], [0.14424768090248108]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.25507786870002747], [0.1776416003704071], [0.0561361089348793], [0.35384827852249146], [0.1015116423368454], [0.23805175721645355]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.0779203251004219], [0.28568974137306213], [0.2259289026260376], [0.08137910068035126], [0.37925606966018677], [0.45194125175476074]], dtype='float32').reshape([6, 1]),
                paddle.to_tensor([[0.4113561809062958], [0.38559842109680176], [0.34662380814552307], [0.25977855920791626], [0.09847179055213928], [0.15297983586788177]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_ac15a258bcd5adf4c7de7ccd0dd92c5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e550c7f7fcb3c155fcac7eb385939c5
        def get_inputs(self):
            return [
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 300, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad1f692859f02aab7389bb31deced781(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 6, 6], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 6, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ddcc54638a18f7ab76b3a022aab1c12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0558386b6be7fef64f0c9e1db5df242e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a20635cdc16ab39bd9c0604e4317ef4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e592e399f288f0cba19158a24e35a337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdb7cdc45aa780613150e23ba695e82f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58ca4942cbd637b17bb3f7b7a798bad1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2777930796146393, 0.02657657489180565, 0.47445645928382874, 0.20409588515758514], [0.1387108713388443, 0.39539310336112976, 0.1251879781484604, 0.12581704556941986], [0.0013369943480938673, 0.06631842255592346, 0.38734182715415955, 0.07199130952358246], [0.4746476411819458, 0.005258830729871988, 0.14967621862888336, 0.14246855676174164], [0.15963207185268402, 0.3332064747810364, 0.09021896123886108, 0.44095081090927124]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_febbde00d51f0611de77faf3e99724ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([5], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_b50880ef6496963befc515fb816d0f46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36710721254348755, 0.2733869254589081, 0.12741735577583313, 0.09983520209789276], [0.1689828783273697, 0.12775754928588867, 0.4929034411907196, 0.024316783994436264], [0.28232342004776, 0.3154520094394684, 0.34447357058525085, 0.1899302452802658], [0.3894905745983124, 0.21266524493694305, 0.15552382171154022, 0.2049335241317749], [0.14210514724254608, 0.42684876918792725, 0.16730007529258728, 0.03494313359260559], [0.36946699023246765, 0.25839653611183167, 0.40660321712493896, 0.049302250146865845], [0.4219869375228882, 0.046291451901197433, 0.45390188694000244, 0.2299237996339798]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0e93612014430188ec26ff867f096481(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a94352e7f9f8e604f4326b6d52eab67b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04564209654927254, 0.4040566384792328, 0.4310336410999298, 0.19761954247951508], [0.06333372741937637, 0.015020381659269333, 0.26050060987472534, 0.4925742447376251], [0.004450720734894276, 0.09676632285118103, 0.15310807526111603, 0.2296193391084671], [0.3995705544948578, 0.20686708390712738, 0.16882404685020447, 0.41516441106796265], [0.10278435796499252, 0.2873406708240509, 0.33401980996131897, 0.45226940512657166], [0.16829141974449158, 0.3355092406272888, 0.43085777759552, 0.3128385841846466], [0.39289021492004395, 0.4308372735977173, 0.11628603935241699, 0.23280969262123108]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cab61b152748f4449b087b0315445390(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55285bb38cf6dcbb857553cd283dcbf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 88, 88], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5514df2ec2b203894d81a721bc832e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.06026938185095787, 0.10351205617189407, 0.010360955260694027, 0.05767056718468666]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_80cb3b8bc81d2ab5df2b5d2fd5811f59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_cdafbdd096b3d7b9e8fea7a9ca85c62c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([300], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7a72843360dfe25ad46d4790396b0e23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96eaa311925eb222c0db43395143a0a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93bbab1863336b191678f9ccedf7249b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93bbab1863336b191678f9ccedf7249b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59fae8a71cb139482d2ac9589425a053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38729b2a3e5ef72bbabab1574afcb2fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e550c7f7fcb3c155fcac7eb385939c5
        def get_inputs(self):
            return [
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 90, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e99bb3ecf818b292ae55f6e1298208e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f454df70ac5248e2da974cdc186ea04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9423ab8ac566408ad429a45a5c39888(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 14, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5153422d562de621eea11544104f82d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7c607b1cc0df38755947f1d23104259(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5153422d562de621eea11544104f82d1
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 144, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7180bf41c7a391ed8e176d6a70eea5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a10b0f6a8251b6f8bef08d88cbb7c3dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a10b0f6a8251b6f8bef08d88cbb7c3dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62ff55f05ef3c28eccc6064cabe177b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3befe3cc79dd7f4776586e7595d7ff6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 17, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e79a67b320f2b957b76171c08cd30f8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 600, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_138ef073ed28995f62bda1e72a45f339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.11586524546146393, 0.3802587389945984, 0.2655188739299774, 0.32558372616767883], [0.2183745801448822, 0.062354110181331635, 0.09305060654878616, 0.19626672565937042], [0.051588743925094604, 0.2933014929294586, 0.06933026015758514, 0.34952929615974426], [0.4719468057155609, 0.09559851884841919, 0.3501664698123932, 0.04593765735626221], [0.3326735496520996, 0.0424429252743721, 0.33680230379104614, 0.1775297373533249]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_febbde00d51f0611de77faf3e99724ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([5], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_642319a6cbb99934b4ea44f577a39c83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 48, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57e9ac54d4341caad3aa7b1ff519dbce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57e9ac54d4341caad3aa7b1ff519dbce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdf351ca22a016a3bdd8985ba0f76d0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e730df7b9c3fdb6eecdc66c804f4c15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adc64b8b9edc4141f2fcb4cdf33a09b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e4b4b3a3c656db94eabcd09610e770a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62ff55f05ef3c28eccc6064cabe177b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9261, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9197b8755d142ebf6105db4792749cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c893ccd2e374042f4dc459a978bfdcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e46fa36943e61e60a370e8494835d0b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 56, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a08f82da31ca65e4cd5496dc9c93c884(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25c90e0a426a6306099b6fc3fad857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a557566363bac992bcd3538bc1f1f2c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bd06f91042fb6cfa1a300ccdb2e5546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e572f7dc80b19dab69cd9ca1965de660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e25c90e0a426a6306099b6fc3fad857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a557566363bac992bcd3538bc1f1f2c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bd06f91042fb6cfa1a300ccdb2e5546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fb2946e234bfa619c2e298006e391f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25348156690597534], [0.1419389396905899], [0.21522848308086395], [0.18140600621700287], [0.35213175415992737]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.47471386194229126], [0.33389967679977417], [0.40402311086654663], [0.4485560953617096], [0.46584224700927734]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.18775731325149536], [0.12440189719200134], [0.05388212949037552], [0.37235766649246216], [0.08130541443824768]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21230792999267578], [0.0743480697274208], [0.0162972342222929], [0.3133012354373932], [0.06640253961086273]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_25faea9d13af884adc9dcd4580c10dfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1649070680141449], [0.39336472749710083], [0.47650161385536194], [0.22819465398788452], [0.16766460239887238]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.21730045974254608], [0.35416045784950256], [0.32481035590171814], [0.011503858491778374], [0.4248878061771393]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.13528534770011902], [0.3804410994052887], [0.13431361317634583], [0.32307666540145874], [0.13232417404651642]], dtype='float32').reshape([5, 1]),
                paddle.to_tensor([[0.4135350286960602], [0.022249840199947357], [0.3079543709754944], [0.28616321086883545], [0.11313261836767197]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_c1da6016d4c431c5a7901c2fe3f9f53c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3226352334022522, 0.1387707144021988, 0.13630889356136322, 0.3679457902908325]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_26b66ad791a671c02914f226ed4f5d82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9197b8755d142ebf6105db4792749cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 96, 144], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c320601e476299e5fe459b24918ad56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d29063b61c5fba7680036558a84fd24c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46b53834586a8897ff3299f17e268f5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76060fe261fcd22420892a75e0ec7de0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 36, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdf351ca22a016a3bdd8985ba0f76d0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 2100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c8647f0f479aac01db243ac8ba39f9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bdb024a8baa0218063cb0424ec2e4db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1730654bff8f7868eab1ed12b9f30e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.4692842364311218, 0.07899530231952667, 0.43798011541366577, 0.0650915876030922]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c747fb5aabfc2d159d84a5bf8277fb67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
                paddle.uniform([15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1917f5ca69fa5739870ab954160a9c03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d3306cb0099b05a8d3569215188c97c
        def get_inputs(self):
            return [
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 112, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c0622d2b202c4abc6635c33c5dc2a29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c0622d2b202c4abc6635c33c5dc2a29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7883a2bf3572fd75101117f35c9dfdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4725, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5424831fc367eb2d46a01852cc857e88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6832866c56cbca1ada98e2d14c7f6f3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_d2b92d8fd6e65b53fbda81105dd128bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_ccb51b0c51f19db2f5bf3a3114a69c51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccb51b0c51f19db2f5bf3a3114a69c51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e572f7dc80b19dab69cd9ca1965de660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9568e8d4f8b80acfdf8352f025921bde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9568e8d4f8b80acfdf8352f025921bde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccfcbbc7044a9666120e0e95456ffeca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 7581, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f525237676a65d6dad9606fa16c5d386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 28, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a85593988e43a6a60cee31d5b9e17811(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_239184908e1d5a33380973011cd34f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([100], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9b0b18056c4f9a90f45fbe88ad3e6e82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b473ed65d58379ab1a4be08fdf7442b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_732976eb00c4f9d7b88746c0b1e6b811(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 11109, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a85149861d1fc45d110ed31465d33a1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 60, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e84c526672e02cb232fc412b97fe03a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2571fe1b9776af245fd1ae82963148fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_419c187bfb52a96b56535834102514ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 120, 200], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46d47c727f77c8a89c9efac48a453a81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a77536c2763d6e001221708b1d5e7298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be31fdabbbe138b4db1f742f529d77cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95229d55329282d2227fd893904126fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4389873175c301ee5bb1370f23ff501(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4b606325eeac1c3290ad672d3a296fa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            return input_0

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc6b9171d05782e4b5cc4a271d23532f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4b606325eeac1c3290ad672d3a296fa
        def get_inputs(self):
            return [
                paddle.uniform([15200, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([3800, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([950, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([247, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([70, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88de75d7bdedc2ebee0e6c7bec03a411(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e48cad4ad4cb5077a9d0e809b734ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed207b7d2541d1cb1d6af37dd414cbad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b95b49ff6eac683b1b6e72a021449ade(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fb4e2110a8574c67d24f7d63466a9b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d813eff4107fbd7e5ae7b21ba3cae4bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5424831fc367eb2d46a01852cc857e88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 12, 12], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88825197c2045f4e3785c418a7d5151e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 12, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_041f26c9e24ebefd28019d743ff37e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_438a389186d9909462da01cb3eed126d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 180, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69338dcf47f4de7d8dbe71e8e6583bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.2400294840335846, 0.33962640166282654, 0.2767137289047241, 0.09159930795431137]], dtype='float32').reshape([1, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_72a08b3e2776683db41fe2c1c7656a49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1e8720eba2ee7060749ba56bcf2cbd7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.43446049094200134], [0.3114164471626282], [0.21863189339637756], [0.22740739583969116]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28952881693840027], [0.0999356284737587], [0.3750646412372589], [0.1684504747390747]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.1857021450996399], [0.232812762260437], [0.44483304023742676], [0.43192264437675476]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.28646060824394226], [0.2347613275051117], [0.07614689320325851], [0.21912746131420135]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_c9e5d00474362dcf69f4216398e0c4f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05875740945339203], [0.3892061412334442], [0.20329052209854126], [0.3478715717792511]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.2514371871948242], [0.428699254989624], [0.0340176485478878], [0.16604164242744446]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.37367141246795654], [0.26348745822906494], [0.22833964228630066], [0.3062015175819397]], dtype='float32').reshape([4, 1]),
                paddle.to_tensor([[0.3861541450023651], [0.4490680992603302], [0.4151049554347992], [0.03166484087705612]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_dac40cbb797b9202b24faff4431a260d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 56, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed207b7d2541d1cb1d6af37dd414cbad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e53989a0d4a0baa10a16a03b6e6c1386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c1d66479b9e4836ca1569fcf1f4a684
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2571fe1b9776af245fd1ae82963148fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4af608d3c6333eebfadb49e3236c1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4af608d3c6333eebfadb49e3236c1c9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59fae8a71cb139482d2ac9589425a053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4116, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_145482f3a29899a037f6c4d66dac8c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 30, 50], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0507d09efa80fc597c786f1140a7f730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 3024, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2909759cccece86a67d494349384d24e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2add71fe7416737b50e0f9c15b6977a7
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196, 4, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee60c9e0572174ead5c95f93343088b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a48c011d1f803e004f2b62c2c1b9bae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_585a26c075ffb2450293d331979e0732(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec15443986f3f393b7b731906147d28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ee60c9e0572174ead5c95f93343088b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a48c011d1f803e004f2b62c2c1b9bae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_585a26c075ffb2450293d331979e0732(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98b18e878b37320d74e6a18bc80a0f7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_027d87f3a989a554a7a90b41e867b074(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2add71fe7416737b50e0f9c15b6977a7
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 16, 12, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 16, 12, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_083ce26d18ee705f657b9da6a599b921(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e79de7cccbbef16c3ceb31c62a033eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 18, 14, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ac295eabc4d4dbe2437f79f9be0fe09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b43407f6e0a8df07d9884ce62e297fe5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b43407f6e0a8df07d9884ce62e297fe5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d85964667a8d1e9e5e605d53f63c4ca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b5fb6a7a4a402beaa8cbc81bbe84d3c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a690b930b90da1666f1f72545f350cb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8afe735d2d65d8469bed35969159bc3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87da7df2ce3d4c8d304e9db0ac2180e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7c504d67503dca941b4c6b128c758d4
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c655ad735407199d9dafad16f8993c1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.13706611096858978, 0.08502600342035294, 0.44341975450515747, 0.3082224726676941], [0.009993945248425007, 0.3372018337249756, 0.11831139028072357, 0.047583818435668945], [0.4995841383934021, 0.01382436789572239, 0.0017661130987107754, 0.08258343487977982], [0.42873167991638184, 0.27258726954460144, 0.20817619562149048, 0.4982690215110779], [0.24455808103084564, 0.42871490120887756, 0.45946192741394043, 0.36610716581344604], [0.05611858889460564, 0.20213836431503296, 0.16467683017253876, 0.09433294087648392], [0.07387426495552063, 0.3788360059261322, 0.16099436581134796, 0.3784846365451813]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_404f9a6587189f18c970129805d0d238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_01dc7c2ae5c4b0af6aefcde3e002efbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5714c09dc0032809f740082851f8d16
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.29233288764953613, 0.36788859963417053, 0.006307649426162243, 0.2807435095310211], [0.3606562316417694, 0.492276668548584, 0.20333808660507202, 0.06060683727264404], [0.14438258111476898, 0.47764408588409424, 0.38392844796180725, 0.3385300934314728], [0.037337061017751694, 0.3944132924079895, 0.4904634356498718, 0.26613932847976685], [0.280810683965683, 0.27498728036880493, 0.3712540864944458, 0.056545354425907135], [0.42267999053001404, 0.3459642827510834, 0.30062058568000793, 0.436982125043869]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
                paddle.to_tensor([], dtype='float32').reshape([0, 4]),
            ]


    class TestPrimitiveOp_fe29c9efde1d0e3e59369ce3065ae09f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c378283a6f7089ddf2b11dab824b1f60
        def get_inputs(self):
            return [
                paddle.to_tensor([6], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
                paddle.to_tensor([0], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2149a5aadec7e0dc831e129dd3661db1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([1, 2, 192, 288], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2da7fa8368a5e2ba9626a133967287b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6e24d39d0bee287db0f1bd336a2ab51e
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 49, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()