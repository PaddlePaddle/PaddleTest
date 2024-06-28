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
    class PrimitiveOp_0f6176f214fecbc1731fbf2ee9f2991e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_564667079f82183d270f09dbb12b1565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f6176f214fecbc1731fbf2ee9f2991e
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c16652e795da5829290bb02325e0c3be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_69b661679b7c4bbc59da2ffeb466ac32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c16652e795da5829290bb02325e0c3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f312701fe71a60882fa1deec9f4d2e3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96258e64ad36cc26f4fd006d7dcbe3d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f312701fe71a60882fa1deec9f4d2e3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1294155a6bfa3f4581261201cd177685(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d779f7f98fcb8c828acd0b4e5b0b845(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1294155a6bfa3f4581261201cd177685
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a639de9c145d62bc20800dce83aec0b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb5199b9a6a09557d79abf4933773492(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a59bb3a3b431cbd51ff708511a8cc2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13ffc323bbacf302746c0377e34c970d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fab08af2e834aa042c3b6e3d13b3bd06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9072bbbc2f49850c39b58e98ad26eb70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd12803bd8bceca937c5940498b2d865(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_95d965159591ec0a8977bf85ce1cb156(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b9b0c16575d837447baa38f1c067ce02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4acffb20df05f1e539694ca168f2a16b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73835792857e26a3f51792962c1d3876(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_342ab0ee1c57f05011e46f42d40a947f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49367632741160b6cf46e6d6e2309ed2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_26fbda7c6b7e7f7e48fcc4e4a85a5607(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49367632741160b6cf46e6d6e2309ed2
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f1fbc347cb347a3e1aa23c73523d2a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f312701fe71a60882fa1deec9f4d2e3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f93de478416d75aa7899a20c09dbb39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_faa34e3c724576d423eeb010174b8d72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49367632741160b6cf46e6d6e2309ed2
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38f274bf4dcdd446c841552a13e0bffb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_755b2a0beab9510828e8cb528b156034(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c16652e795da5829290bb02325e0c3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d9bd02e1e358ed4f0e25aee772db8b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f5e8db66212543225a51c254d19a26f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ec06169424e4fd617c5f3b4c0eb43f68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f5e8db66212543225a51c254d19a26f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e805769e3f84222fd9ce6fcf58a6fcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75ac7e02952e6e75ff4abd4d435a0815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ac7e02952e6e75ff4abd4d435a0815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e077ef29f57ed52f9a4ccca58b5d973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e077ef29f57ed52f9a4ccca58b5d973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e09e5c8049ebfcad9e3d55a95a865b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e09e5c8049ebfcad9e3d55a95a865b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a8266a8301390cd3867d530f5a47021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a8266a8301390cd3867d530f5a47021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3fec043736db47dc569951d0283dad42(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [], False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3501ff9e4c7855e9cbea669bc472e933(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fec043736db47dc569951d0283dad42
        def get_inputs(self):
            return [
                paddle.to_tensor([9.641922950744629, 1.3869916200637817, 1.2816085815429688, 1.134856104850769, 1.710841417312622, 1.9208142757415771], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0ed95c5156b553be682704c514538a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ed95c5156b553be682704c514538a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0f0e42e7905afe8a171fdcffad2f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0f0e42e7905afe8a171fdcffad2f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f66e845ae561ea051dd82fc91ce570bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f66e845ae561ea051dd82fc91ce570bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b48268ecddec254ae356bfb309df9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b48268ecddec254ae356bfb309df9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_84f1201394199131bb351f016c387384(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd537aedea8df2d5d0f56abba4919133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f1201394199131bb351f016c387384
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94cba21b9d711dcf6d557d253ac3397f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_398ad5f2b7ec518551226d286a3ff57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efe6c66e92045f741301864ebb945908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b075b59cd91bae357d070659aca3ccfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f312701fe71a60882fa1deec9f4d2e3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ed95c5156b553be682704c514538a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ed95c5156b553be682704c514538a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0f0e42e7905afe8a171fdcffad2f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0f0e42e7905afe8a171fdcffad2f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f66e845ae561ea051dd82fc91ce570bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f66e845ae561ea051dd82fc91ce570bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b48268ecddec254ae356bfb309df9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b48268ecddec254ae356bfb309df9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e5f8476e191383379999322fd9e583a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4725765743dea4c82e8a6c25ad4100bb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44d7d60d71a5139ab5dded2efc18dc68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4725765743dea4c82e8a6c25ad4100bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ccfb09cdc5f7f461a9a80e4067d4a6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c16652e795da5829290bb02325e0c3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a348348347a75ef5e5841d489a3760(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a70f6ecb8297474e0e369b724040fed3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0f08023ae600a4fd5e6f511742bb120(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a70f6ecb8297474e0e369b724040fed3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bcd24deb7987377a893de5c264e5c77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a2b769b9ee297d1797f49344d663abc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c16652e795da5829290bb02325e0c3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2973e267eb032a7a4ca063962241673(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39e733af98ad50b8df908ebe1b6d2fe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_019aa8915d9981e0a25d796d08be8e87(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_820fee57c163123aa33c975f87da4447(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_019aa8915d9981e0a25d796d08be8e87
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6cc9440509ebe8a9ca97a24805f58672(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d1afb6bf6f31e5d723e42dac587fc8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_722f64902e2f38b27dfc9d4ba7bb2a4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f6176f214fecbc1731fbf2ee9f2991e
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c77b58e0565ce7230930e2c213521a06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f1201394199131bb351f016c387384
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dbb5965edcff0b5bc4f0f60031dcfaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44cf8351e2654bc5cbe869028e7ce8ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d92e04e81b8c51e4d837b08a663a137f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17db583e7e73a1b3edae7ef4ba14e775(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ee66e601aa792de3082add6a4ec796a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_70139440dc727df1211fc7f47ad3130a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee66e601aa792de3082add6a4ec796a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3ca72357cba7cd7ce2fde6003e4cb91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74544e1796e5353f52d01a85fa05bca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f312701fe71a60882fa1deec9f4d2e3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ff605ac41cef242fbadcdfb2d60cd0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4725765743dea4c82e8a6c25ad4100bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a221326eb5537ca33fdfa371bb18eac4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c16652e795da5829290bb02325e0c3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ac7e02952e6e75ff4abd4d435a0815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ac7e02952e6e75ff4abd4d435a0815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e077ef29f57ed52f9a4ccca58b5d973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e077ef29f57ed52f9a4ccca58b5d973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e09e5c8049ebfcad9e3d55a95a865b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e09e5c8049ebfcad9e3d55a95a865b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a8266a8301390cd3867d530f5a47021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a8266a8301390cd3867d530f5a47021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1a7dbb60932f8e0783eb5e641c8bc83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f312701fe71a60882fa1deec9f4d2e3d
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4289e5be39ff95c6bde5495aad346197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_083a51d05a904d9d8b09e82d5b1cdbf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_368c95891ae62c797b0061fb9ee08169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0fe8c1c4ab58302a254974ab9a09596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_334beee4135c1d35018d035a84dc493c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cfca80734f4e1cea4b7a75c1248959d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca2db54840e4a5b1d075e0fa3459fc4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_627961cf2cbbfb9f9ed7d9654e40cf82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13fe1ad8ed95c5e8c842bb93a7ebfa99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19b8d79f12b829ea93096f706f2feced(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_610f3af154f202f634f750dfdd125073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19b8d79f12b829ea93096f706f2feced
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e69e6314bb7eac9486e2d6218669761f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a639de9c145d62bc20800dce83aec0b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e895801f5fa230702407905a8354e4f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4725765743dea4c82e8a6c25ad4100bb
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_390c8c47443bfb8247d698656739c3a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee66e601aa792de3082add6a4ec796a9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f77ae0ec6ee3295349d96419d7ef43df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c16652e795da5829290bb02325e0c3be
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b5ea74ffec4810c63530cb9733410b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f1201394199131bb351f016c387384
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f3e506e4700a19fa06c67d1c1d7df74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49367632741160b6cf46e6d6e2309ed2
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bffd8a62792ccd25428cd5da99ea0b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd741d0450ffaf962b5b46dc06e99786(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aac73e27c0175066427397edaf73e0ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1294155a6bfa3f4581261201cd177685
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bd656f59ca5e632986d397969c8356b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_019aa8915d9981e0a25d796d08be8e87
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c80f51712c07aca82b9f8d21bb8508f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22117b3abfe90e08f0c56b64bd86862a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2519114fd5b804b6222a489d4143ab1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f5e8db66212543225a51c254d19a26f
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f25f060998715b511206e60f0ed592aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa15ee6a1c808c800e7c808ca423c09d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c097c29dc87eb83d1fcc5aabdb4bcdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa15ee6a1c808c800e7c808ca423c09d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9266c1bddfda4762c6ca2f61ca5c5a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5412f27c7e0b9189d8e265b67d4fecdf
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_541ea901f5937cbe7f39387fdff1b81a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d3993e36fa77190b53aff8d0a9c0d19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_541ea901f5937cbe7f39387fdff1b81a
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_945f6aeab687d204de17c34b322af52c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b9b0c16575d837447baa38f1c067ce02
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_03ca4cbdf230e43eb601eb817b5a6aea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4, 4, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2bce4a73da69de496299dfb12a6ddbb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03ca4cbdf230e43eb601eb817b5a6aea
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82f7e227b2a304e9bb62f3732f54b3f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1294155a6bfa3f4581261201cd177685
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f9a6979bba1df60f229d650e4ad0a871(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dde7fd395f03caefad2a57ca182be6d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9a6979bba1df60f229d650e4ad0a871
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_010c6bada9101f3420bd05bdd6928137(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 152, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_29532112201e974b8a1d17d93ceeb633(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_010c6bada9101f3420bd05bdd6928137
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_48c0de0fbb13d7154ca3cc6dc3eed80b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 768, 13, 13], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad093c98a2432d36794fe5677d491c21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_48c0de0fbb13d7154ca3cc6dc3eed80b
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a24ffefc432334fe0335201d36660297(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 104, 104], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_91a738360e0b4501e09ddfefefb2ac24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a24ffefc432334fe0335201d36660297
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_84f4d2a00f7ecd761122d2e85829bd14(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 17, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da3732d2415abf895666baf60cb57b28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_84f4d2a00f7ecd761122d2e85829bd14
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_511a8d3c698cdb628bd4679472af68d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 48, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1e05d2ea062734d20f2e95d6ba43e40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_511a8d3c698cdb628bd4679472af68d5
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ee1d97ea1cafbf38c553f7ef4f6eb0a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b218f069f251540c9f07b06bb488e37d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ee1d97ea1cafbf38c553f7ef4f6eb0a
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4efbba1635e373ce3e6402a67801423(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 112, 112], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2eb96f76d0f7acce7c2c48160a1b3d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4efbba1635e373ce3e6402a67801423
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e953d03549bfac659f33d358daea1acc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6485094e03bd2c97c8ad322c69386533(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e953d03549bfac659f33d358daea1acc
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e09a167183b7a55ef1dd65426eae6c9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 34, 34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8896241179e70c6992a01badece47f2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e09a167183b7a55ef1dd65426eae6c9a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_28f20ddc8dd6482212f0d8095f986dc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ebdce46b90d2ce07337006fc98acc43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28f20ddc8dd6482212f0d8095f986dc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_341fccb689f36f20726bfa4ac7f1da79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_34d7ae0fa04c97bb852a8c06969d67c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_341fccb689f36f20726bfa4ac7f1da79
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_55b97b3be4ff4b198f2977bee88926c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 13, 13], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba551487ca86740f58551832aea24b5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55b97b3be4ff4b198f2977bee88926c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dcfac29b8e323c0d2b347993519d24aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_800d6e1cbcaac0800626b8a1283d8820(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dcfac29b8e323c0d2b347993519d24aa
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f34e6404fd7b02bad775adfcaf9e66c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 576, 13, 13], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57f5be9076cbb9aa1d71df267f71ecbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f34e6404fd7b02bad775adfcaf9e66c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_621d49ae71f2cb9fed1bc0fe9ab4c1f8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 768, 12, 12], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2667c5cf77d58d268fec676634e96509(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_621d49ae71f2cb9fed1bc0fe9ab4c1f8
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1624aea75f346e7d00e88044c1f63e2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9141424f542b5b15272ee81353902f7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1624aea75f346e7d00e88044c1f63e2f
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b142808a66ed25aa2d6eabf4898d055(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 576, 15, 15], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d2f8c507153572afbc694c2d0bd4053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b142808a66ed25aa2d6eabf4898d055
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34b50e933f79a23ca7d9443818b083af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 10, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90179ce888a855a43c0c7bddfe6b05a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34b50e933f79a23ca7d9443818b083af
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6506bbd130537966716c02fe918cd5b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 160, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_505486f7e7464e8094b6c4556107b99a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6506bbd130537966716c02fe918cd5b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b1c77aaae75596eeaf6bbe7b1f947a95(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_049ca9c2d5111e9c4c7d5ee07703b844(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1c77aaae75596eeaf6bbe7b1f947a95
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e62f270a377b0453670bbbb7ee3b20a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 168, 168], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3633b3f92f7a13db0a15aa73c416167(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e62f270a377b0453670bbbb7ee3b20a4
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a7367d26649f1326da463731da6a3f5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_159b0ff7b325e18fda59bd391126ec21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7367d26649f1326da463731da6a3f5c
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_af5291cdf018a0977394577e57f673cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 64, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e697ff78e473d846821ad8e610a8209d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af5291cdf018a0977394577e57f673cc
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e697ff78e473d846821ad8e610a8209d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af5291cdf018a0977394577e57f673cc
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5a61392d5741563565c0ffd2df834960(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 128, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_301b85f811792407a72c30d8e8cc6339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a61392d5741563565c0ffd2df834960
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_301b85f811792407a72c30d8e8cc6339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a61392d5741563565c0ffd2df834960
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_585d12ae68fd396cbdedb4e22e5119e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 256, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c24eb281482cfa9a4a2437390a67c322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_585d12ae68fd396cbdedb4e22e5119e5
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c24eb281482cfa9a4a2437390a67c322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_585d12ae68fd396cbdedb4e22e5119e5
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cda87b3a541faf1fdb700883a99341a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 512, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebfecc50774ceea846345021e2feb7a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cda87b3a541faf1fdb700883a99341a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebfecc50774ceea846345021e2feb7a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cda87b3a541faf1fdb700883a99341a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da38a039be00f8413a2ec1e5df06853a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [], False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_808108da47331a61458e19a688944580(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da38a039be00f8413a2ec1e5df06853a
        def get_inputs(self):
            return [
                paddle.to_tensor([9.641922950744629, 1.3869916200637817, 1.2816085815429688, 1.134856104850769, 1.710841417312622, 1.9208142757415771], dtype='float32').reshape([6]),
            ]


    
    class PrimitiveOp_6922af13bb76f42243f0a4ab25e0fa11(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 64, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c5dc4d01ef4f9cede2a3af37ba98320b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6922af13bb76f42243f0a4ab25e0fa11
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5dc4d01ef4f9cede2a3af37ba98320b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6922af13bb76f42243f0a4ab25e0fa11
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45be518864fd817b3de5f8861cad44a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 128, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_890874afd13d14bdca5d2e9143cfa9a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45be518864fd817b3de5f8861cad44a5
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_890874afd13d14bdca5d2e9143cfa9a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45be518864fd817b3de5f8861cad44a5
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_33c45bd67edced84ba00ba3cc5d6c63f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 256, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_233a354dcdd0bf7e8f0947efc15d4e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33c45bd67edced84ba00ba3cc5d6c63f
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_233a354dcdd0bf7e8f0947efc15d4e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33c45bd67edced84ba00ba3cc5d6c63f
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_806912e46efb9c347275fd15ba539710(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 512, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36c6edf14c9941ffbec059439da123ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_806912e46efb9c347275fd15ba539710
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36c6edf14c9941ffbec059439da123ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_806912e46efb9c347275fd15ba539710
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_320307c551c6e838992ef47c01a4e751(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 52, 52], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a4b41ce308dc4e0b9592e3c305c862f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_320307c551c6e838992ef47c01a4e751
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b742e130566007d8b56016b1f2125a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d66400284755ed9105c0524112f0399(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b742e130566007d8b56016b1f2125a7
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2fa399b618f5c9d822ba0a3a18e57050(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 76, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f0f3869fb61394a7f095b0b851fd1204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2fa399b618f5c9d822ba0a3a18e57050
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3f871306a601819c93032adefbd8a483(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f3ac15f7199acb81dd7f24543e7e2a9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3f871306a601819c93032adefbd8a483
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c748f5eb2371aae937c11ac09ff861c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 768, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78440be2ec919fb1c1c00f6a23bfbce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c748f5eb2371aae937c11ac09ff861c
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5dc4d01ef4f9cede2a3af37ba98320b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6922af13bb76f42243f0a4ab25e0fa11
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5dc4d01ef4f9cede2a3af37ba98320b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6922af13bb76f42243f0a4ab25e0fa11
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_890874afd13d14bdca5d2e9143cfa9a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45be518864fd817b3de5f8861cad44a5
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_890874afd13d14bdca5d2e9143cfa9a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45be518864fd817b3de5f8861cad44a5
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_233a354dcdd0bf7e8f0947efc15d4e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33c45bd67edced84ba00ba3cc5d6c63f
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_233a354dcdd0bf7e8f0947efc15d4e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_33c45bd67edced84ba00ba3cc5d6c63f
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36c6edf14c9941ffbec059439da123ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_806912e46efb9c347275fd15ba539710
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36c6edf14c9941ffbec059439da123ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_806912e46efb9c347275fd15ba539710
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a41694c54fa87dba7b0349a07a90ba61(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_befaede23c670c386cd512e30c8b08f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a41694c54fa87dba7b0349a07a90ba61
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6fb8b188148b95eaa5c5c0e7bc2081cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_151c29d6f0dda75a5519d839cc001840(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fb8b188148b95eaa5c5c0e7bc2081cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ee292e85428e88ef507415243bb257cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 136, 136], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a14fffed360470e60999aa2f562e980e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ee292e85428e88ef507415243bb257cb
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db1371ad57d37456350bf6aede186c56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d43dffc18fd7be011be14ad8d2754904(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db1371ad57d37456350bf6aede186c56
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0f08023ae600a4fd5e6f511742bb120(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a70f6ecb8297474e0e369b724040fed3
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2555924cd9752557959324e86f3a49e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_16be7088cbe1f8942afe260c7e5d6b53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2555924cd9752557959324e86f3a49e3
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_87011921f7bf538708d5cecc2040ab8a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 104, 104], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41e22356f0b188bfe94bcf9f54f9ca24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87011921f7bf538708d5cecc2040ab8a
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a0383b31670b56652f88d5c12e169de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 184, 184], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_13de2e09805f469564fa1b80c152725e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a0383b31670b56652f88d5c12e169de
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_94e46d88d09db2018a82a7a292bb2a32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 52, 52], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32fd85333fdc72ec5726cdc53ccf9e70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94e46d88d09db2018a82a7a292bb2a32
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3c5d929ecdc43d8451d0afa74064cc76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_466c467219c3b56056fd20acdc909cb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c5d929ecdc43d8451d0afa74064cc76
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c312cfaef6501ae8e1194cd31940e92d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 104, 104], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b10da4e75ef57fdef38529d99419f882(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c312cfaef6501ae8e1194cd31940e92d
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_91b795456f55095ad4960b84fa44d912(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd80d68d92e3e4f16be19be16cced2d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_91b795456f55095ad4960b84fa44d912
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7ae81a52639d2920ed3e6867dfde62f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 480, 42, 42], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c95fc604a92ae9e4d413a42ddb0fd175(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ae81a52639d2920ed3e6867dfde62f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a6f9c8a3b7a02dffa407b27478a114d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 60, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d2fd5f502b4a346566a822d4819a51a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a6f9c8a3b7a02dffa407b27478a114d
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1959a450376f34028c0b6426e0e9de55(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30daf018eff9985ec491caddbe7cacd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1959a450376f34028c0b6426e0e9de55
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4af0e49894cf6e69c92b9bb6f88c8139(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 19, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d132b897d60baac4f823049bcc4c3bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4af0e49894cf6e69c92b9bb6f88c8139
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dbc15942888eb19e92f49b6a67e2f973(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_494862d683c8d9567b0a75b0aeb8c5ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbc15942888eb19e92f49b6a67e2f973
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_757e3162fb597aeda1362d7c197c4e99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e37a8ac0ac4cb53f8683ac2a13a382a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_757e3162fb597aeda1362d7c197c4e99
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bca5fb8cf38a9a7c37c256c1496277a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d923d7f559f006a4a4ecdbf5d4a7cb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bca5fb8cf38a9a7c37c256c1496277a7
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e38a6bf0e245452c4d1abd94ebccf00d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_409eaaf8662fa71245c11ab599df9cfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e38a6bf0e245452c4d1abd94ebccf00d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_69b489b5485d931b49627e4d2afff678(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 768, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a317f5c2cf0dabfc6b04297d6ec7e539(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_69b489b5485d931b49627e4d2afff678
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_66887929dabe0f3a0d2ab92b4366114b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 30, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ff3f62df88208d2eb9c8b6173aa098ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_66887929dabe0f3a0d2ab92b4366114b
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e7932bfd718d3bc2796182227b264009(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3f431bfda16d5e771c507fa115d7bc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e7932bfd718d3bc2796182227b264009
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e697ff78e473d846821ad8e610a8209d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af5291cdf018a0977394577e57f673cc
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e697ff78e473d846821ad8e610a8209d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af5291cdf018a0977394577e57f673cc
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_301b85f811792407a72c30d8e8cc6339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a61392d5741563565c0ffd2df834960
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_301b85f811792407a72c30d8e8cc6339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a61392d5741563565c0ffd2df834960
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c24eb281482cfa9a4a2437390a67c322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_585d12ae68fd396cbdedb4e22e5119e5
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c24eb281482cfa9a4a2437390a67c322(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_585d12ae68fd396cbdedb4e22e5119e5
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebfecc50774ceea846345021e2feb7a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cda87b3a541faf1fdb700883a99341a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebfecc50774ceea846345021e2feb7a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cda87b3a541faf1fdb700883a99341a2
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b4f7cde286f3bbf04cabd54f761ed423(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 768, 23, 23], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_948c3d550a74334b708f3d1449c6d9a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4f7cde286f3bbf04cabd54f761ed423
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b578ee881dfd21df651603bb5b118433(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aeb64f92349f6a7c5239a2d6dd062a58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b578ee881dfd21df651603bb5b118433
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86c8b3c3a7bc986cdfaafcaca468f73d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_233ed2ae8bc2003efc9fba0f4db19d27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86c8b3c3a7bc986cdfaafcaca468f73d
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4e3dc6ddcf914e13437e5308c9d122e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d27d21eaa3699feaee844818b58e6971(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e3dc6ddcf914e13437e5308c9d122e3
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ffb8406516e09c5be7ad07180c2894e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54a0da030769c65703c99f2528c39760(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ffb8406516e09c5be7ad07180c2894e3
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_964ade48d336c950bcd64677de11d007(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 92, 92], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_537eff2252c426cd8b25967f1624d604(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_964ade48d336c950bcd64677de11d007
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e0a593084672a834f7d305bb7089d37b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4db1bf63462e1469fa68808c76af9cc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a593084672a834f7d305bb7089d37b
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ccb772f9bc06ae12d9f21b90c2fec4d7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 52, 52], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f132f9f21cf345f7ce16af0f450042ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccb772f9bc06ae12d9f21b90c2fec4d7
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f2c4486790cf803651d2e39262d263ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c98f39eda0e7f3a6152efa78be6efea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2c4486790cf803651d2e39262d263ad
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f1cc4575eea35cb1279d35a67ebb544c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94f36ce559f7333d1f1264ddca62c672(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1cc4575eea35cb1279d35a67ebb544c
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_610f3af154f202f634f750dfdd125073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19b8d79f12b829ea93096f706f2feced
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ecc739c0a1221331f325d410ba6160d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 384, 46, 46], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f79b1df5c8319059c3cf93a56eeaf22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ecc739c0a1221331f325d410ba6160d
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_45d15dba5cc404fe4429f2a1b4c44498(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 288, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab8748243b69a650e498d17ace0c0205(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_45d15dba5cc404fe4429f2a1b4c44498
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_772c9a2d1f98391ba8d18aa79b112c56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 240, 84, 84], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4732bcbb3a6dab570c39c9f396fe1f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_772c9a2d1f98391ba8d18aa79b112c56
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f43e5bb6e61f762d4aabfbf6f459b19(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 48, 80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9612903eabc8a8289d15f5caaba8f7a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f43e5bb6e61f762d4aabfbf6f459b19
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f6b7df8d111fed4e21059e4f2554098a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 144, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_85324a2ef13da21682d0fef2e48a333d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6b7df8d111fed4e21059e4f2554098a
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0e153147a6ce09ce093e678e7cac3c74(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 576, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d5b97c07aa4b4c874cd3a261019e79a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e153147a6ce09ce093e678e7cac3c74
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4dd6a5593126af8c158c6b84b8d8b87a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 26, 26], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c067cb8dbe3734ad04f43de5ccce2e8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dd6a5593126af8c158c6b84b8d8b87a
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a68069c873be9544221528e752825a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_112ce5bbbb0b2fce51715aad4969be4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a68069c873be9544221528e752825a2
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_38bcd95dfc0037029cca046c71ec69bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 120, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47b2f4d9a03940d59686fea8a797c45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38bcd95dfc0037029cca046c71ec69bd
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d60e74a4d07d03db51db2e2daf7b33d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 960, 21, 21], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ee8a44a79b339f9cd6386032cde32e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d60e74a4d07d03db51db2e2daf7b33d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17f0e63b3ca1c819d3db74e91565d9fb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 192, 38, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d054bfd2ecc4ea067464b523face2210(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17f0e63b3ca1c819d3db74e91565d9fb
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_21bb6a92743d03369e121c9c770f7ced(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 120, 256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f596d6e19f66ff53c1cdff0edb81011b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_21bb6a92743d03369e121c9c770f7ced
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_88e2ff4581bcb62b60f3dba053fa7bcd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 68, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e6439d4a4d2ba841e8e12e804502e70c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_88e2ff4581bcb62b60f3dba053fa7bcd
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c097c29dc87eb83d1fcc5aabdb4bcdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa15ee6a1c808c800e7c808ca423c09d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4fe330a80a6deb4cfe12851ef32bcf2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c61dcf849ff515d1727b6f39df428799(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4fe330a80a6deb4cfe12851ef32bcf2f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d3993e36fa77190b53aff8d0a9c0d19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_541ea901f5937cbe7f39387fdff1b81a
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7e3155f51052f5c0c80c925def40dc0d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e7ab4fbd894465d41057d3ed63f30f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e3155f51052f5c0c80c925def40dc0d
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2bce4a73da69de496299dfb12a6ddbb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_03ca4cbdf230e43eb601eb817b5a6aea
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0fa9a0127a105d9c935cb73259a1aaeb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 72, 256, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0edd1748921edf72ee217fd83ebc69c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fa9a0127a105d9c935cb73259a1aaeb
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2, 3], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c7100e050236f8a883350df1f90b3f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccabc5fc8133f6dc614542cbfded0c2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 152, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c507b17f3837bb71091b820f78746e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbfbca09f4e7ebbc015fb76e7263a1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86749b780bf23d5c189f5f3ce862b4d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_804157cf1deae535dffecc103c73ccce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee228fb65ca727644487f70876609119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d497dee072fe39c252b6a8f0ddb4bf9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3db7965e9f7d6a963c535c2e6cf7562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7e7f399a45c6c07238fad0e7560c560(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d06e52d11d7e3e8794b46a47145440f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [-1], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5fbe060d5762e1ffead4eb91b9bc815e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7eba69063bdbd7112fdce2fd5591d4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd9acb5e39069844834e118771d95087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d73d3f4145038ecf1e447e333559f38f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42af500f4d0717888a69021553b0c60d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b945b159b3ec58edf77771ab58b4cd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8305c818576833adb0df7e6b6b779fcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca81d9d45333c1079f6f32b4df720f3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c4737563d88212186b2f5140f46dd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62eda26eb4065e2fe36184176ccfaf8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc7d8c006d97697e806f8d73825f78fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 168, 168], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d238bc13cfc6c15e0c6bc0084310f01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ac7e02952e6e75ff4abd4d435a0815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ac7e02952e6e75ff4abd4d435a0815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e077ef29f57ed52f9a4ccca58b5d973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e077ef29f57ed52f9a4ccca58b5d973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e09e5c8049ebfcad9e3d55a95a865b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e09e5c8049ebfcad9e3d55a95a865b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a8266a8301390cd3867d530f5a47021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a8266a8301390cd3867d530f5a47021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3501ff9e4c7855e9cbea669bc472e933(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3fec043736db47dc569951d0283dad42
        def get_inputs(self):
            return [
                paddle.to_tensor([9.641922950744629, 1.3869916200637817, 1.2816085815429688, 1.134856104850769, 1.710841417312622, 1.9208142757415771], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_0ed95c5156b553be682704c514538a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ed95c5156b553be682704c514538a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0f0e42e7905afe8a171fdcffad2f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0f0e42e7905afe8a171fdcffad2f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f66e845ae561ea051dd82fc91ce570bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f66e845ae561ea051dd82fc91ce570bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b48268ecddec254ae356bfb309df9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b48268ecddec254ae356bfb309df9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90debdea1261c10beba4a58e383469a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8e7ed68187c9901ddf8cb47b0478304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92942085c5702c6c7fdc1b27d85bf4d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c973893ef5ff8874c25e38f7eabee3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_99b98f392d9316f4aef4669504934204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ed95c5156b553be682704c514538a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ed95c5156b553be682704c514538a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0f0e42e7905afe8a171fdcffad2f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0f0e42e7905afe8a171fdcffad2f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f66e845ae561ea051dd82fc91ce570bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f66e845ae561ea051dd82fc91ce570bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b48268ecddec254ae356bfb309df9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b48268ecddec254ae356bfb309df9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bb6e8df20f7ad403c7c1e0279da62d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c8dfc3fe9fd10856de813da9fcca3f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77074abd88f4abbcd6d55117671c5d2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 136, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54bcc0985131d2c651177d31dedc8e87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57774a6d1f167b0683b7d39c19164aef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean(input_0, [2], True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0bc888d514918089adf65e68c7dd2217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57774a6d1f167b0683b7d39c19164aef
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09437922b30f014412e84714e1e7d57c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e3675bace69707a3eef60b0f0264643(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0880c8dac02653637b7829c75aea2c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 184, 184], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b251203ad60497fca39271507192a0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ef5256acc8aaf0eeae02ace886891c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f6ded44fb94f311d4f1ecfcb1c46f61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 104, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b60db0e3aa114fd55e6e37e43b8b1d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc07a262d0f63ace9b608bd13dcd546f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe52fd6ab4f807be3a82d1a90a7f1217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_347584b0d383e8e1eecb450c7fe7ce99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b686240eb5ffd159950a92a495fb903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48da960896863808d44150fb79e84585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7952107f119f3081532e897516251b4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c21ca9342d2c5ed0032396e89f42ba34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a79381322479813d591b6e335b9eff55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7f44b042c53384b926927cd1333592d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a869f44ad690393f52238478a839ed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d25af60de091852b5cf55a5cd79b50e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ac7e02952e6e75ff4abd4d435a0815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75ac7e02952e6e75ff4abd4d435a0815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 64, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e077ef29f57ed52f9a4ccca58b5d973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5e077ef29f57ed52f9a4ccca58b5d973(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e09e5c8049ebfcad9e3d55a95a865b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2e09e5c8049ebfcad9e3d55a95a865b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a8266a8301390cd3867d530f5a47021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a8266a8301390cd3867d530f5a47021(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf96fad883112b7b4ce34f3e0c01773
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a1ac7db69fe889ee73266b5a332aa431(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d12d04104826786d142cd04ca25d728b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9d4eb1ad8c55692863710261bdd3087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f503f1f8e440db5b607a8d62ce24861f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f40bd5d1bb4097b0f2f367c69e6f01b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55f42166950aca80cdb24947f72a92b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57781d5f8594ac6a61eddada56e4c708(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3edd960851fc2eee8ad06f6900643afe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ce0bd4006309e25e285ca697683873e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cecdb6ff9c2d4e4b665f4f795f722f0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f3721fe1e086f648c74fefe0cf0e3f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57774a6d1f167b0683b7d39c19164aef
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d554dbc8c756c055f1d103e629450ece(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd40219f4276c9a15aeb2a1ae6c908aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b59b04e7ef610ee1c30b09e86528f9dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9fa1a09ddb3e0f5ae897be429712023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 48, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d77fa15546f47fec1e4d0fc20c6a61a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b131d234b58e9442d265254d977dee1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_737c51cc37e39cc0da34bd4c065ba956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5498c275ed0f354f87ea53c14c3d9353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c4ff9357cdae6c588df2b65a5cdc280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e86eede46f6c8150c4e48fcc3b9b8902(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81c364effca643fafd33cf491069f6a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5ec0fa5c7936338c401abb4f2e19855(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 120, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8320f3e795aa2511fe33e7d3d2484f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff7e8e52082ac3c06376d3d0e1aa299c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57774a6d1f167b0683b7d39c19164aef
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_753e53652bc951b7667d08df9e6ae608(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_070ee4885598f2ef5169aa8c443cabab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57774a6d1f167b0683b7d39c19164aef
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0afeaa88e675de6df485b4a76c1a2abe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa9c4bacc4bdcc673109c33b8bb7ceb2
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_211fe2869655e06de9891e3e51848918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57774a6d1f167b0683b7d39c19164aef
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 4, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e13df98f7fec32e46af079767359a01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b048ed4b44e49c6d8c37653d03f22c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()