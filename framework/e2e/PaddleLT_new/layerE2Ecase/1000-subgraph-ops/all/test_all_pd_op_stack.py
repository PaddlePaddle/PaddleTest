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
    class PrimitiveOp_07408811cc46c2bd6fb31783328bb32d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3843dbb5a8ac836fe2e3c56f02c6514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(36, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_62ef1835293491d42af515f2c12491ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_fa934cdbd01e3f5c483486416b159c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(3549, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(19, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_3b19964477096fe05027dfc769e542d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eb9663974b686d5a013d34aac1ac9651(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2):
            input_0 = [arg_0_0, arg_0_1, arg_0_2]
            return paddle._C_ops.stack(input_0, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4]
            return paddle._C_ops.stack(input_0, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c09e63a73aa876cf6cc4acf5ceb8861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(20, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3a604c36450e84f952a1bc1994740347(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(40, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5cdfc1227bd7208167ef905aa34657e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(40, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_0c4455c133d7e6903bd79eba1c47303e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(80, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_75e940e80a93496e11fef6970abc8ac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1a08aabe4f54cf5b0c93c809bd3c83c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_6672a0410ad5f47dec5453373a5ca7a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a221ff75f7b70cfc4c6a26fc5a32e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6672a0410ad5f47dec5453373a5ca7a7
        def get_inputs(self):
            return [
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_776ed9b4bd2a61fb86b1b1021d0cd97f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
                paddle.static.InputSpec(shape=[48, 48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f83d44bcf01c6df9d33fa8922f19bb43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_776ed9b4bd2a61fb86b1b1021d0cd97f
        def get_inputs(self):
            return [
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fcebbcf0dbb3f88ed2e1fc2d68b44f08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47c18f11024218914f97fce4654129d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcebbcf0dbb3f88ed2e1fc2d68b44f08
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a221ff75f7b70cfc4c6a26fc5a32e12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6672a0410ad5f47dec5453373a5ca7a7
        def get_inputs(self):
            return [
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f83d44bcf01c6df9d33fa8922f19bb43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_776ed9b4bd2a61fb86b1b1021d0cd97f
        def get_inputs(self):
            return [
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47c18f11024218914f97fce4654129d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcebbcf0dbb3f88ed2e1fc2d68b44f08
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8b32b7149711b43d3f05c08bdbdad3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9b7a4d1c54e45fafb963aa9cd314c121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_33a64ccd71286ecba146be2a60487982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(21, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5]
            return paddle._C_ops.stack(input_0, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f5d93e45b6ea1e77d368189c3898abb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(13, dtype='int64').reshape([]),
                paddle.to_tensor(13, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1b38de09bff6f64bb760f79de3e12a71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(7581, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7a5712b78aa3088b9f10af1e793c700e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_440eeb466448dadcd8d46d3fdde318d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_92c2903e34e1a9a4a34dcbd3eda1b8a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

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


    class TestPrimitiveOp_4a057908b2b04c6c22b98873ec73e0ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92c2903e34e1a9a4a34dcbd3eda1b8a5
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d72963778bf70d801b9f20a612062345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(25, dtype='int64').reshape([]),
                paddle.to_tensor(38, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_598d49bd4458bcc3ef584303ff9e2196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1bca38664400a9d38bd88cabd582486f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c8a0ee399479c069711ff80c3265b943(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b06021a78ba140beed14f82ad1461f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(4725, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_87f2aee83ebcb05787767dd65989b8b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_87c01a4f698f698f9d57c0749090ed5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(577, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3010d76d2b945d4aeebe26f481873a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(577, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_94c9d6c42cdf3f1e9af24f5663a6b547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5d2053018ea4d9214bf5d09f7c899456(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_834aa833882cb7745156925da9bce46b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_2ef978b76c3b8ef83a09b86eb68ab337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_35a99cf3a9fa4b0bd13375ca769793c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_34f93f07c6993ccde1d989f6e3db4f47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8630e8c1e91d38964c51beaf57ca5bdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f27276eb450eeafb8255252183092945(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(8400, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9c7db4c9cf4c8093ec150452070853da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(20, dtype='int64').reshape([]),
                paddle.to_tensor(30, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7dc8f01733416d671850494031f8296a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_49c363060c417ec2efda56e3a1a2c9b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1a1004ede43d8dff9fcde0865e0f1c01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49c363060c417ec2efda56e3a1a2c9b6
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_689f2cc9f800eac4ca4e77391b07f6f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11ea28fcdba8591316d4cc28304dadc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_689f2cc9f800eac4ca4e77391b07f6f5
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_55e5ed4541622e45ec8a568204dddb91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
                paddle.static.InputSpec(shape=[16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b77df1fe0ebd17b422084a7e494d834(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55e5ed4541622e45ec8a568204dddb91
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1a1004ede43d8dff9fcde0865e0f1c01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49c363060c417ec2efda56e3a1a2c9b6
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11ea28fcdba8591316d4cc28304dadc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_689f2cc9f800eac4ca4e77391b07f6f5
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b77df1fe0ebd17b422084a7e494d834(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55e5ed4541622e45ec8a568204dddb91
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_11912c77d30c9ba4e22e47e7028fe4f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9b7a4d1c54e45fafb963aa9cd314c121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_710179b00c86cab68631501e7a294071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(3549, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_75e940e80a93496e11fef6970abc8ac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3218bac4c017393849412d580f9c66fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_553a6cbd5fa2e38f4ea5ba615fcc8da4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5836cc735bc9bab1305d5480dbe7c6ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_d4de41e0861d2b98d56dbb0074aa13b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int64'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61161e9ee424d848ff513e121b6af21b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4de41e0861d2b98d56dbb0074aa13b5
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7c581687580a7352258b0bbfc3dd4b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4de41e0861d2b98d56dbb0074aa13b5
        def get_inputs(self):
            return [
                paddle.to_tensor(98, dtype='int64').reshape([]),
                paddle.to_tensor(99, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ab438fe2ecd9e9fda28568a3ef045813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_509e57355b58b11438f6201fc84c3768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_cf1d1872efa44acdc3fd2c6069f089d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1280, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d672997b9682e9429f81bf474ba352f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7cc80e8fb99d8662c4e2f1ba6302f427(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5dee1f92d4f18d87d0ea814cb536516e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(19, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_df846741a52bebddb7db9ca80dc3685d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9cc1c80b3239a333f83006405aa3896c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a1ae66553523b1c356159682b1ac46a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9264739ada0068f085a89e31ed80cd93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_598d49bd4458bcc3ef584303ff9e2196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1bca38664400a9d38bd88cabd582486f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_eb4805e59f5331eb371e6a8a4f1523e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_487e459d979c760d2046afd80d40c41f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_24957f75a709d9f776aaec4f1f5d4c26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3218bac4c017393849412d580f9c66fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_00829c58a1f1f596269be911366e01ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d9e393b7644a8d8ea5db3858d509f805(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_01f279527e3683534b52096c252e48b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ddd0873a98a6572ae35f51a96534af34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_aec105cc597b76c431f98cc3bbf77289(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
            return paddle._C_ops.stack(input_0, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 49], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4fe8aa85b44c697f793741391fc3d72d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aec105cc597b76c431f98cc3bbf77289
        def get_inputs(self):
            return [
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6ca8f25073d67208157c7165956b804(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
            return paddle._C_ops.stack(input_0, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 784], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2892ea20e47a399f473f6eec7c3a3e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6ca8f25073d67208157c7165956b804
        def get_inputs(self):
            return [
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cdcdc5da65846e28c2e0c7439d951ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7cdcdc5da65846e28c2e0c7439d951ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9b84d82ab3fa6d0b93c371aeca411482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_90b90e52b889dd21aa92c951a4cc9746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ace0d6ee0f113627e98e19ad1d46315b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(36, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(50, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3a1a536bbb0d68949842ae6c81a7aae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(72, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(50, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_74877f47d68af8fc8e37a8f40c578179(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(4116, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9cdc72aa9b0eb6713cbc1b8fa612a9dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_4ce2c8c3c9e506188372212803f20d1c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[80, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee8fb69cc5da69dc9b5c38dd018def8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ce2c8c3c9e506188372212803f20d1c
        def get_inputs(self):
            return [
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e1eeb04e2c8449039b8ee09fb421a5e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
                paddle.static.InputSpec(shape=[40, 40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d89188df8ded995cfdd0582e9f41c955(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1eeb04e2c8449039b8ee09fb421a5e4
        def get_inputs(self):
            return [
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f095fb96227c043a755e19065bc2721(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[20, 20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90273d46125445bdfdf44702bb318e0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f095fb96227c043a755e19065bc2721
        def get_inputs(self):
            return [
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee8fb69cc5da69dc9b5c38dd018def8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ce2c8c3c9e506188372212803f20d1c
        def get_inputs(self):
            return [
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d89188df8ded995cfdd0582e9f41c955(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1eeb04e2c8449039b8ee09fb421a5e4
        def get_inputs(self):
            return [
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90273d46125445bdfdf44702bb318e0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f095fb96227c043a755e19065bc2721
        def get_inputs(self):
            return [
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b7a4d1c54e45fafb963aa9cd314c121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_30d639388bf106de4ba457d135d7bb4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
            return paddle._C_ops.stack(input_0, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
                paddle.static.InputSpec(shape=[22, 196], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15c9126468df09a0728264a9dbe7bade(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_30d639388bf106de4ba457d135d7bb4c
        def get_inputs(self):
            return [
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dc8f01733416d671850494031f8296a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_81a2e25959ee586ab259093334cfa989(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(6069, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4a5b952ad5b48d6c184612c673e1114a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8d629960b268f1a7d8d6984118bb744c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f18ace1fad98b046cb3dec9850022bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_01743a8b961ddfa71cd50cc829c3e0cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(52, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(202, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_36da9898e51c26991072a60fd4599257(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(52, dtype='int64').reshape([]),
                paddle.to_tensor(202, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_cc321821191b77a8b53f20d18b962e14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1025, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a54cc9f0e6f04202739cac6a36814959(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1025, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bbc948755a7a3249c76d4d5dc49bd01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4bd332b4d9db706cd815a81a5b88b783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(150, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_786f2ff917b69ce1d6f912837a3d3025(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_936792fd4e76ca98ea0454a68fa83826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_786f2ff917b69ce1d6f912837a3d3025
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a38ebdb3089f53519c47dcfa34781340(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d53ed2a2a23ec694eb324740d3bfcf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a38ebdb3089f53519c47dcfa34781340
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0ccc7f5a06899ba2cb27efdbe148447(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b804088bffc671f64b3960e66870d403(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0ccc7f5a06899ba2cb27efdbe148447
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_185d7563572d8382e1a2b0bef0a14182(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ac1a67ab8165068767f826707ea2154(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_185d7563572d8382e1a2b0bef0a14182
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85bfe53d9e0e8c04424c7be8fe046e68(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f965448caea910ee7ad2ea8a7aaac84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85bfe53d9e0e8c04424c7be8fe046e68
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd199401886eab3cf604c22da9fa91fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e3aa3161820706b871079feec924005(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd199401886eab3cf604c22da9fa91fe
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c267e343bac3ef16277484f33e527ded(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d17502d008f0c7b32f9935780e79b4a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_94c9d6c42cdf3f1e9af24f5663a6b547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_197b11883028e9da3574616e596405b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(13, dtype='int64').reshape([]),
                paddle.to_tensor(13, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3712b6b3b900acdfa47c6f9c634be4f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(104, dtype='int64').reshape([]),
                paddle.to_tensor(104, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3ce919760667e19649ffcf75b01b112c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9e9404db8f67a3d51691e8f5b4e76ee4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_487e459d979c760d2046afd80d40c41f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_0c69e90751d0664694901ba6391e91df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_e39ad08ee135ebcb0b35a8a78cc81ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_fff67b560be2f7c3e86352207ca43614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(232, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba3301fcf2b8fd58b3d223431ab9795c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(464, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8e675dde0523b376a114f1755de77803(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_42d9b8b63d80f7e73d8f39ed1ede28a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_e1a213f3a8d5404173694ad5130d3983(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_80a2932be1d5b5c257f902cca9d7c146(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bfe6843c790a099d18cd092bb159e8e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5927abd103098a33b96ab65c9845b40e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bcf8671c4aebb4d771f2cfcee3836709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9395683607c7322d3bbdc034e962d14f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(72, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(25, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6ee798c22fc052a88f26cb82560862f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(144, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(25, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b7e9782f3b43e804bfadf83d680c462f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d3294120cf5e5cd7b09c8f5c0be5188d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3891e7d078922bdde0e56c3dc7017f16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4bd332b4d9db706cd815a81a5b88b783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(150, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a3ee016c4b5fe985d9848db5b1030304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(9261, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d8ec883fdf4d290d6cbbd2ab43869d64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_28d41cae8420befef8268682111a508d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_23a7ea94154ceaf65c16e8f57220af5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6ae0d3e1e50331774c3531f96d175cae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_c4588cf4b9f638a7e7f76e8ccf73a575(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
                paddle.static.InputSpec(shape=[68, 68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac983746d84f7b4de10b3c69f6d94ecb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4588cf4b9f638a7e7f76e8ccf73a575
        def get_inputs(self):
            return [
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cb5db160e12cffcca8b6c261d4f1f61e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[34, 34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84c75447654c2a201c8b6246fa3e573a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb5db160e12cffcca8b6c261d4f1f61e
        def get_inputs(self):
            return [
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_00e65c6f3307905a7fcce81544c734ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
                paddle.static.InputSpec(shape=[17, 17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82b25ec9782cfab907aed64f875b5067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00e65c6f3307905a7fcce81544c734ef
        def get_inputs(self):
            return [
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac983746d84f7b4de10b3c69f6d94ecb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4588cf4b9f638a7e7f76e8ccf73a575
        def get_inputs(self):
            return [
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84c75447654c2a201c8b6246fa3e573a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cb5db160e12cffcca8b6c261d4f1f61e
        def get_inputs(self):
            return [
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82b25ec9782cfab907aed64f875b5067(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00e65c6f3307905a7fcce81544c734ef
        def get_inputs(self):
            return [
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4fcd26b4798ffe5c590ceb0066341cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2048, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4a5b952ad5b48d6c184612c673e1114a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_56f4a25deede1a59991be5734990d6a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_423c3aa46214addc277341434905b6ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_739e2bc051c0a68fc210aecfeaa1c133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_666abb421e25c26e0500c4255baad95e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721105f32334966636a6f874a138dbf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ced4b13084128ecd7c3d7b531d2a51a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(320, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5a549b73918367ba2db3f999aa8aa85f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2100, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b8b722fe9a8b9be9b15b9b693b1fedc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4fcd26b4798ffe5c590ceb0066341cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2048, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bfe6843c790a099d18cd092bb159e8e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d82ad92b070e62c113b06062f42312d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(15, dtype='int64').reshape([]),
                paddle.to_tensor(25, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_94c9d6c42cdf3f1e9af24f5663a6b547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_c0bf6516a66945064410a0349df0f781(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
            return paddle._C_ops.stack(input_0, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
                paddle.static.InputSpec(shape=[10, 784], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e20ca2a512b7913e78ea80837946284a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0bf6516a66945064410a0349df0f781
        def get_inputs(self):
            return [
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dea35c40d3bf2c726bc559486d43fc1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1025, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6df39bdee8561a889a9abd96d3bfbb3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1025, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_87ae3257f86cf85cfd756cdb60142625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f625b964a697823b4dcb9ec61cba0e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d3294120cf5e5cd7b09c8f5c0be5188d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b8b722fe9a8b9be9b15b9b693b1fedc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9e2e24a70b5b4601ef4b1743b6fc14f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a03f45ada5fcf014b8dffc7dc363fb9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d03150255f80cc12f68a0f207c8df6e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_588653eaa3e4658ffab6ad8855b13116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(11109, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8118f354c9d1fd86dbb40b4e9c4d73e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1280, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_e8b32b7149711b43d3f05c08bdbdad3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_509e57355b58b11438f6201fc84c3768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c953181114b13c8fff2b5da579c343d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f4a7b57c02f2e50a5a8d111b97f9b386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(77, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_42bb30ddea29f6b43e87df03867cf3f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0ffbb2921b1b3673927293b304337bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_42bb30ddea29f6b43e87df03867cf3f9
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d89f87fecbab979fc6bfe44da36fed5a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
                paddle.static.InputSpec(shape=[100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c9f82ebecc564787fe274314c84568b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d89f87fecbab979fc6bfe44da36fed5a
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_25806e320056ca1f16d1719001f821b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e7ee4f141f0812ae1378fc1ff7e3903(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_25806e320056ca1f16d1719001f821b3
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c72ce37f31c3c74a4fa45fedfe74589(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
                paddle.static.InputSpec(shape=[50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_90cb4fa99e54154d05900ef039d2dcce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c72ce37f31c3c74a4fa45fedfe74589
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d05153d756026f5649bb7e4b4d5ec5c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e7a07085a1ef961b7a37b9cfc5046012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d05153d756026f5649bb7e4b4d5ec5c1
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e4ce82ee83e102e0adb1b0c6fd5ff09c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
                paddle.static.InputSpec(shape=[25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9cac49ddb15ec207a9f0069338e0e7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4ce82ee83e102e0adb1b0c6fd5ff09c
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8989361ae00e914ff866fa058a4539d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c23974065aa6f9c2fb05db2052b213b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8989361ae00e914ff866fa058a4539d6
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe918b3a626a1cabe9b3f643dbf7277c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
                paddle.static.InputSpec(shape=[13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fdd3da73818bc03eb9a6d68c3f662163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe918b3a626a1cabe9b3f643dbf7277c
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_599e0f1232e3cb5b647de3aa66e0b03b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9740ccb0152e76e112f4718312d7ea94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_599e0f1232e3cb5b647de3aa66e0b03b
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7e66250bc2bfde96e4cc39dafd11d9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
                paddle.static.InputSpec(shape=[7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6239c4cac1c82621d455f31a2dd734ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e66250bc2bfde96e4cc39dafd11d9a
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cdc72aa9b0eb6713cbc1b8fa612a9dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7cdcdc5da65846e28c2e0c7439d951ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6730ccf7cdf56bd9aa370a06128e2e0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ffa6ae11382462b4ab96f6d4c9a651e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9fac50951826d3b26991d30ea3577178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(320, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b648c2cc6735791a9bc9b6c6693265d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9c09e63a73aa876cf6cc4acf5ceb8861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(20, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3a604c36450e84f952a1bc1994740347(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(40, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5cdfc1227bd7208167ef905aa34657e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(40, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_0c4455c133d7e6903bd79eba1c47303e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(80, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_01d2e16527182c88af5927d71db08947(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(80, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3ce919760667e19649ffcf75b01b112c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ad87571aea4161c8aead692c43a33821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a03f45ada5fcf014b8dffc7dc363fb9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_530b11ec3f927a822fae086461a92c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_35a99cf3a9fa4b0bd13375ca769793c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_75e940e80a93496e11fef6970abc8ac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3b52c5ca12efc2a924a7bb75e98cb6be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(116, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9b85c47dff2a2b703b6defe69bdb44fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(232, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7dc8f01733416d671850494031f8296a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b38e44f32137e3ddf5e737d45eb785d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_24d49931fb8a1506f0269cdedf0f612e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_46e103587b0bc8be1c155bde1c87f00c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(320, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_42d9b8b63d80f7e73d8f39ed1ede28a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bbc948755a7a3249c76d4d5dc49bd01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_aa23079cb17e8d66949b11216a7ed11a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(320, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bfe6843c790a099d18cd092bb159e8e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c1fcafe971f5e1dfad5145808cb1cde5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(3024, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_38675725191ae6c4eaa4e4c3e8eaa2ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
                paddle.static.InputSpec(shape=[72, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bdb4a2d7eb3f332e61c175ef789f30d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38675725191ae6c4eaa4e4c3e8eaa2ef
        def get_inputs(self):
            return [
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e868363a62eef4e91284989796106140(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
                paddle.static.InputSpec(shape=[36, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1db002e8472676e2327d47332dbf4bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e868363a62eef4e91284989796106140
        def get_inputs(self):
            return [
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3a74fc604efec0e9cf8566b06ddbd9a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.stack(input_0, -1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
                paddle.static.InputSpec(shape=[18, 18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d44acd19702948080223625e38e720cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a74fc604efec0e9cf8566b06ddbd9a7
        def get_inputs(self):
            return [
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdb4a2d7eb3f332e61c175ef789f30d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38675725191ae6c4eaa4e4c3e8eaa2ef
        def get_inputs(self):
            return [
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1db002e8472676e2327d47332dbf4bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e868363a62eef4e91284989796106140
        def get_inputs(self):
            return [
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d44acd19702948080223625e38e720cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a74fc604efec0e9cf8566b06ddbd9a7
        def get_inputs(self):
            return [
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5e59e59b7841754d313c97e8fdb128c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1174, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_914dfec9c2b929b1494309eee1b5cd05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1174, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1afc243f81fabdbb6f44378919129519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(150, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b343b0016319d4dfea12afb2cc0af9c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9cdc72aa9b0eb6713cbc1b8fa612a9dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_0834889a45402f8f473f780af038262d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(58, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_be6c4252154a457981a0b75e46cdb362(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(116, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_2d0355dce7944242cdaf0466a7620d25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1174, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_244448fa593074e87224e1d501566c7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1174, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b7bb4c58facadb724f2651cd0d81e4c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d3294120cf5e5cd7b09c8f5c0be5188d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d3843dbb5a8ac836fe2e3c56f02c6514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(36, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_62ef1835293491d42af515f2c12491ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_fa934cdbd01e3f5c483486416b159c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(3549, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(19, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9c09e63a73aa876cf6cc4acf5ceb8861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(20, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3a604c36450e84f952a1bc1994740347(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(40, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5cdfc1227bd7208167ef905aa34657e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(40, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_0c4455c133d7e6903bd79eba1c47303e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(80, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_75e940e80a93496e11fef6970abc8ac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1a08aabe4f54cf5b0c93c809bd3c83c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3ac25f52889830a33b919c00036ead32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1bb729504090310effb57dc44551554(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2f087ea87b4ebc2c2890cbf18a6d739(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3ac25f52889830a33b919c00036ead32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1bb729504090310effb57dc44551554(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2f087ea87b4ebc2c2890cbf18a6d739(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8b32b7149711b43d3f05c08bdbdad3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9b7a4d1c54e45fafb963aa9cd314c121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_33a64ccd71286ecba146be2a60487982(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(21, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5f5d93e45b6ea1e77d368189c3898abb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(13, dtype='int64').reshape([]),
                paddle.to_tensor(13, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1b38de09bff6f64bb760f79de3e12a71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(7581, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7a5712b78aa3088b9f10af1e793c700e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_440eeb466448dadcd8d46d3fdde318d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_d6e8d3ee4404c3b7d4e9a7b8d3df7f16(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

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


    class TestPrimitiveOp_9df91466ad053ef8c240de6b154b3d60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d6e8d3ee4404c3b7d4e9a7b8d3df7f16
        def get_inputs(self):
            return [
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
                paddle.uniform([300], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d72963778bf70d801b9f20a612062345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(25, dtype='int64').reshape([]),
                paddle.to_tensor(38, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_598d49bd4458bcc3ef584303ff9e2196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1bca38664400a9d38bd88cabd582486f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c8a0ee399479c069711ff80c3265b943(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b06021a78ba140beed14f82ad1461f79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(4725, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_87f2aee83ebcb05787767dd65989b8b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_87c01a4f698f698f9d57c0749090ed5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(577, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3010d76d2b945d4aeebe26f481873a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(577, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_94c9d6c42cdf3f1e9af24f5663a6b547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5d2053018ea4d9214bf5d09f7c899456(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_834aa833882cb7745156925da9bce46b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_2ef978b76c3b8ef83a09b86eb68ab337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_35a99cf3a9fa4b0bd13375ca769793c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_34f93f07c6993ccde1d989f6e3db4f47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8630e8c1e91d38964c51beaf57ca5bdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f27276eb450eeafb8255252183092945(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(8400, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9c7db4c9cf4c8093ec150452070853da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(20, dtype='int64').reshape([]),
                paddle.to_tensor(30, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7dc8f01733416d671850494031f8296a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_423c3aa46214addc277341434905b6ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_423c3aa46214addc277341434905b6ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_11912c77d30c9ba4e22e47e7028fe4f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9b7a4d1c54e45fafb963aa9cd314c121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_710179b00c86cab68631501e7a294071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(3549, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_75e940e80a93496e11fef6970abc8ac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3218bac4c017393849412d580f9c66fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_553a6cbd5fa2e38f4ea5ba615fcc8da4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5836cc735bc9bab1305d5480dbe7c6ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_61161e9ee424d848ff513e121b6af21b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4de41e0861d2b98d56dbb0074aa13b5
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7c581687580a7352258b0bbfc3dd4b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4de41e0861d2b98d56dbb0074aa13b5
        def get_inputs(self):
            return [
                paddle.to_tensor(98, dtype='int64').reshape([]),
                paddle.to_tensor(99, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ab438fe2ecd9e9fda28568a3ef045813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_509e57355b58b11438f6201fc84c3768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_cf1d1872efa44acdc3fd2c6069f089d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1280, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d672997b9682e9429f81bf474ba352f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7cc80e8fb99d8662c4e2f1ba6302f427(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5dee1f92d4f18d87d0ea814cb536516e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(19, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_df846741a52bebddb7db9ca80dc3685d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9cc1c80b3239a333f83006405aa3896c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a1ae66553523b1c356159682b1ac46a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9264739ada0068f085a89e31ed80cd93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_598d49bd4458bcc3ef584303ff9e2196(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1bca38664400a9d38bd88cabd582486f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_eb4805e59f5331eb371e6a8a4f1523e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_487e459d979c760d2046afd80d40c41f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_24957f75a709d9f776aaec4f1f5d4c26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3218bac4c017393849412d580f9c66fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_00829c58a1f1f596269be911366e01ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d9e393b7644a8d8ea5db3858d509f805(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_01f279527e3683534b52096c252e48b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ddd0873a98a6572ae35f51a96534af34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_27909b64250083b60264c67428679ed3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3, arg_0_4, arg_0_5, arg_0_6, arg_0_7]
            return paddle._C_ops.stack(input_0, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e65be94d81d107f5ee08ced0d4826b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27909b64250083b60264c67428679ed3
        def get_inputs(self):
            return [
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 49], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2db603866211ad813858627498bc7728(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27909b64250083b60264c67428679ed3
        def get_inputs(self):
            return [
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cdcdc5da65846e28c2e0c7439d951ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7cdcdc5da65846e28c2e0c7439d951ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9b84d82ab3fa6d0b93c371aeca411482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_90b90e52b889dd21aa92c951a4cc9746(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ace0d6ee0f113627e98e19ad1d46315b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(36, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(50, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3a1a536bbb0d68949842ae6c81a7aae0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(72, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(50, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_74877f47d68af8fc8e37a8f40c578179(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(4116, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9cdc72aa9b0eb6713cbc1b8fa612a9dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7463dcdef5106be11b18a6e97768c657(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2d13247cf0c27a4435a31a4d772bc94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95fe4ad3b29292b766e2e0c962d91f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7463dcdef5106be11b18a6e97768c657(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2d13247cf0c27a4435a31a4d772bc94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e95fe4ad3b29292b766e2e0c962d91f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
                paddle.uniform([20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b7a4d1c54e45fafb963aa9cd314c121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_0a9613f38ab7bdb13aacdadad4dcd25a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27909b64250083b60264c67428679ed3
        def get_inputs(self):
            return [
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
                paddle.uniform([22, 196], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dc8f01733416d671850494031f8296a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_81a2e25959ee586ab259093334cfa989(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(6069, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4a5b952ad5b48d6c184612c673e1114a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8d629960b268f1a7d8d6984118bb744c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f18ace1fad98b046cb3dec9850022bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_01743a8b961ddfa71cd50cc829c3e0cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(52, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(202, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_36da9898e51c26991072a60fd4599257(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(52, dtype='int64').reshape([]),
                paddle.to_tensor(202, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_cc321821191b77a8b53f20d18b962e14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1025, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a54cc9f0e6f04202739cac6a36814959(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1025, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bbc948755a7a3249c76d4d5dc49bd01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4bd332b4d9db706cd815a81a5b88b783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(150, dtype='int64').reshape([]),
            ]


    
    class PrimitiveOp_2d319dc3ba109a036ada4fd34b1306a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1, arg_0_2, arg_0_3):
            input_0 = [arg_0_0, arg_0_1, arg_0_2, arg_0_3]
            return paddle._C_ops.stack(input_0, -1)

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


    class TestPrimitiveOp_90d27cd513076d6d7b6f08f4b112f64f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d319dc3ba109a036ada4fd34b1306a1
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b6db31197fe11daa2b739a6d41458d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
                paddle.uniform([14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c80035cb888769cb977f669c506afd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d319dc3ba109a036ada4fd34b1306a1
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b268e592f273e7785ee93b1599fbe1fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
                paddle.uniform([28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34894506b6297c90ca186449142d616a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d319dc3ba109a036ada4fd34b1306a1
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f69959e94c4af593bec2670e56aac5af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c267e343bac3ef16277484f33e527ded(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d17502d008f0c7b32f9935780e79b4a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_94c9d6c42cdf3f1e9af24f5663a6b547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_197b11883028e9da3574616e596405b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(13, dtype='int64').reshape([]),
                paddle.to_tensor(13, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3712b6b3b900acdfa47c6f9c634be4f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(104, dtype='int64').reshape([]),
                paddle.to_tensor(104, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3ce919760667e19649ffcf75b01b112c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9e9404db8f67a3d51691e8f5b4e76ee4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_487e459d979c760d2046afd80d40c41f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_0c69e90751d0664694901ba6391e91df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_e39ad08ee135ebcb0b35a8a78cc81ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_fff67b560be2f7c3e86352207ca43614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(232, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba3301fcf2b8fd58b3d223431ab9795c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(464, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8e675dde0523b376a114f1755de77803(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_42d9b8b63d80f7e73d8f39ed1ede28a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_e1a213f3a8d5404173694ad5130d3983(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_80a2932be1d5b5c257f902cca9d7c146(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bfe6843c790a099d18cd092bb159e8e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5927abd103098a33b96ab65c9845b40e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bcf8671c4aebb4d771f2cfcee3836709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9395683607c7322d3bbdc034e962d14f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(72, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(25, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6ee798c22fc052a88f26cb82560862f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(144, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(25, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b7e9782f3b43e804bfadf83d680c462f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d3294120cf5e5cd7b09c8f5c0be5188d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3891e7d078922bdde0e56c3dc7017f16(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4bd332b4d9db706cd815a81a5b88b783(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(150, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a3ee016c4b5fe985d9848db5b1030304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(9261, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d8ec883fdf4d290d6cbbd2ab43869d64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_28d41cae8420befef8268682111a508d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_23a7ea94154ceaf65c16e8f57220af5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6ae0d3e1e50331774c3531f96d175cae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_815c3a18038ee0e251544e5f78f24b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9f7a5e1b506c96cb724a3bddc4a8277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8114d6490240d6d46b950b6bf75ebbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_815c3a18038ee0e251544e5f78f24b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9f7a5e1b506c96cb724a3bddc4a8277(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8114d6490240d6d46b950b6bf75ebbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
                paddle.uniform([17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4fcd26b4798ffe5c590ceb0066341cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2048, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4a5b952ad5b48d6c184612c673e1114a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_56f4a25deede1a59991be5734990d6a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_423c3aa46214addc277341434905b6ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
                paddle.uniform([16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_739e2bc051c0a68fc210aecfeaa1c133(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_666abb421e25c26e0500c4255baad95e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
                paddle.uniform([8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_721105f32334966636a6f874a138dbf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ced4b13084128ecd7c3d7b531d2a51a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(320, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5a549b73918367ba2db3f999aa8aa85f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2100, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b8b722fe9a8b9be9b15b9b693b1fedc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_4fcd26b4798ffe5c590ceb0066341cb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2048, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bfe6843c790a099d18cd092bb159e8e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d82ad92b070e62c113b06062f42312d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(15, dtype='int64').reshape([]),
                paddle.to_tensor(25, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_94c9d6c42cdf3f1e9af24f5663a6b547(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_dc91139809d8e8dc6cb1d14b3d75d8bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27909b64250083b60264c67428679ed3
        def get_inputs(self):
            return [
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
                paddle.uniform([10, 784], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dea35c40d3bf2c726bc559486d43fc1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1025, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6df39bdee8561a889a9abd96d3bfbb3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1025, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_87ae3257f86cf85cfd756cdb60142625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f625b964a697823b4dcb9ec61cba0e7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9d844fdad0a62c7ab08408e985ecac6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d3294120cf5e5cd7b09c8f5c0be5188d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b8b722fe9a8b9be9b15b9b693b1fedc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9e2e24a70b5b4601ef4b1743b6fc14f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a03f45ada5fcf014b8dffc7dc363fb9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d03150255f80cc12f68a0f207c8df6e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_588653eaa3e4658ffab6ad8855b13116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(11109, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_8118f354c9d1fd86dbb40b4e9c4d73e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1280, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_e8b32b7149711b43d3f05c08bdbdad3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_509e57355b58b11438f6201fc84c3768(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c953181114b13c8fff2b5da579c343d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(7, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_f4a7b57c02f2e50a5a8d111b97f9b386(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(77, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_00c7dd2eadb830e81f408f62f742f9ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d319dc3ba109a036ada4fd34b1306a1
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a23187d7217c0da2c7b3a589875c7ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
                paddle.uniform([100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b2e33380eb73b33be26603b6bd63212(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d319dc3ba109a036ada4fd34b1306a1
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e79e30fba67bbcb67fa070aed97e9c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
                paddle.uniform([50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f934e257274582b7ee6e624649d2417a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d319dc3ba109a036ada4fd34b1306a1
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9700da362a27d9428ab1b859fcf4ded(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
                paddle.uniform([25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cd36350f82c1c4ee332d9abc5ab70ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d319dc3ba109a036ada4fd34b1306a1
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe86fce2818b2dc3ae82ed592265fdb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
                paddle.uniform([13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26929b246e334fdb02df9d380494e467(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d319dc3ba109a036ada4fd34b1306a1
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7086173177c90d13fe556068036e70b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
                paddle.uniform([7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cdc72aa9b0eb6713cbc1b8fa612a9dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7cdcdc5da65846e28c2e0c7439d951ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6730ccf7cdf56bd9aa370a06128e2e0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a3ef07bf0ae4f3719f947faba3c01205
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(24, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ffa6ae11382462b4ab96f6d4c9a651e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(48, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9fac50951826d3b26991d30ea3577178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(320, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b648c2cc6735791a9bc9b6c6693265d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9c09e63a73aa876cf6cc4acf5ceb8861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(20, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3a604c36450e84f952a1bc1994740347(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(40, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5cdfc1227bd7208167ef905aa34657e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(40, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_0c4455c133d7e6903bd79eba1c47303e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(80, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_01d2e16527182c88af5927d71db08947(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(80, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3ce919760667e19649ffcf75b01b112c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ad87571aea4161c8aead692c43a33821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_a03f45ada5fcf014b8dffc7dc363fb9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_530b11ec3f927a822fae086461a92c95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_35a99cf3a9fa4b0bd13375ca769793c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(28, dtype='int64').reshape([]),
                paddle.to_tensor(192, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_75e940e80a93496e11fef6970abc8ac3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(-1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_3b52c5ca12efc2a924a7bb75e98cb6be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(116, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9b85c47dff2a2b703b6defe69bdb44fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(232, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_7dc8f01733416d671850494031f8296a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ba734761f1658f73ad0985f4c4aeb19a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b38e44f32137e3ddf5e737d45eb785d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_24d49931fb8a1506f0269cdedf0f612e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_46e103587b0bc8be1c155bde1c87f00c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(320, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_42d9b8b63d80f7e73d8f39ed1ede28a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bbc948755a7a3249c76d4d5dc49bd01f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(0, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_aa23079cb17e8d66949b11216a7ed11a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(320, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_bfe6843c790a099d18cd092bb159e8e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(14, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c1fcafe971f5e1dfad5145808cb1cde5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(3024, dtype='int64').reshape([]),
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(17, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_139d5fbb727135cfd3e6e8203a57fa44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17ecd9da0d42399b592a908c77394ea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aef1ae3fc2c98e099b0c54a69f767f83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_139d5fbb727135cfd3e6e8203a57fa44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17ecd9da0d42399b592a908c77394ea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aef1ae3fc2c98e099b0c54a69f767f83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
                paddle.uniform([18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5e59e59b7841754d313c97e8fdb128c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1174, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(6, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_914dfec9c2b929b1494309eee1b5cd05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1174, dtype='int64').reshape([]),
                paddle.to_tensor(384, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_1afc243f81fabdbb6f44378919129519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int64').reshape([]),
                paddle.to_tensor(256, dtype='int64').reshape([]),
                paddle.to_tensor(150, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_6a61d1a02fdf9a20995de9928bcb3b2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b343b0016319d4dfea12afb2cc0af9c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
                paddle.to_tensor(16, dtype='int64').reshape([]),
                paddle.to_tensor(32, dtype='int64').reshape([]),
                paddle.to_tensor(160, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_9cdc72aa9b0eb6713cbc1b8fa612a9dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_0834889a45402f8f473f780af038262d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(2, dtype='int64').reshape([]),
                paddle.to_tensor(58, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_be6c4252154a457981a0b75e46cdb362(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(116, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_2d0355dce7944242cdaf0466a7620d25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f948553ebe75d770b9cfe07e77313f6b
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1174, dtype='int64').reshape([]),
                paddle.to_tensor(3, dtype='int64').reshape([]),
                paddle.to_tensor(12, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_244448fa593074e87224e1d501566c7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(-1, dtype='int64').reshape([]),
                paddle.to_tensor(1174, dtype='int64').reshape([]),
                paddle.to_tensor(768, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b7bb4c58facadb724f2651cd0d81e4c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(64, dtype='int64').reshape([]),
                paddle.to_tensor(128, dtype='int64').reshape([]),
                paddle.to_tensor(512, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_614ee47c3924ce31684fb807c91c165e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfc15afee627836777dd8953d000ae31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(1024, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_c3bb7931e06b20ddb17be3bc3aea731b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec0602271123a693d6458f3dc9e3443f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(4096, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_b3824876dc6a2a2c49f33e0754a63944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b19964477096fe05027dfc769e542d3
        def get_inputs(self):
            return [
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82d81b39e6888b9d98e4bcb823199fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb9663974b686d5a013d34aac1ac9651
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int64').reshape([]),
                paddle.to_tensor(16384, dtype='int64').reshape([]),
                paddle.to_tensor(1, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_d3294120cf5e5cd7b09c8f5c0be5188d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07408811cc46c2bd6fb31783328bb32d
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(56, dtype='int64').reshape([]),
                paddle.to_tensor(96, dtype='int64').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()