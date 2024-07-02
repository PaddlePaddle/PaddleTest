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
    class PrimitiveOp_1915df63be2b42d85e07e3a18fe68cb9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9fed2bc5d8c0fa2a3b6146a5a8f3aea4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1915df63be2b42d85e07e3a18fe68cb9
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a62a8add873717be8aff7961246f2078(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 92], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cabfd47878501989615b9df0f1dd28a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a62a8add873717be8aff7961246f2078
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4af816ac33ab3b06ade811caeebb984d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df9489db0d89eb59882d05c1563cdcad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4af816ac33ab3b06ade811caeebb984d
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_04cf1499a215f4d8cc7b20e1004e075b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c5b1084a2f4343eaa6b03284d8abb382(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04cf1499a215f4d8cc7b20e1004e075b
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_00ffcd55bcef04018aa126132337de44(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -10000000000.0
            input_2 = 4.135169982910156
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ddd2a898f7d3cbbafb7ab4091c1c2d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00ffcd55bcef04018aa126132337de44
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4248183071613312]], [[0.1123322993516922]], [[0.19716881215572357]], [[0.3563118278980255]], [[0.3588925302028656]], [[0.3417365252971649]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 3.402820018375656e+38
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7f4aca9fb890defdceddbc5ad1f1b0d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 336], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e84c6fd1d15005e3e4e63e2d3e6c693b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7f4aca9fb890defdceddbc5ad1f1b0d
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d1cfc8524a72d202e95e716d4a37b773(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 60], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a896085b66c9edbb236bb92e53f1a209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1cfc8524a72d202e95e716d4a37b773
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1fc7e45470ecbec44798a02656659898(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1.0
            input_2 = 3.402820018375656e+38
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a0d075d8b2e69d6c36713c27486df7f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fc7e45470ecbec44798a02656659898
        def get_inputs(self):
            return [
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_6dd4d03fb3b0d76b94848279ddb38be8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7f4aca9fb890defdceddbc5ad1f1b0d
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dd4d03fb3b0d76b94848279ddb38be8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7f4aca9fb890defdceddbc5ad1f1b0d
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a0e64daab47f18d2ce4978eb638cfb1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb21b85655e87544ea47f39ed7b4d052(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a0e64daab47f18d2ce4978eb638cfb1
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7852f0141a5428cdaa0ff8c18f7d054f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1cfc8524a72d202e95e716d4a37b773
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2632b854a8aa16888ff566b5f76f56cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 872], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c136bd5c1dd47bc8549b5aa16aadc9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2632b854a8aa16888ff566b5f76f56cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 3.402820018375656e+38
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4cbe426e2da424e4e6626309a7076fb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cbe426e2da424e4e6626309a7076fb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 15.989999771118164
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4912c94c350068f05fb5a29229270d00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abbcd6ca5b7042d0bca00b6c89c06632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7f4aca9fb890defdceddbc5ad1f1b0d
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 3.402820018375656e+38
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_baeb93708d0e6e194a5bae54cff3268c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2542710304260254], [-0.1396092176437378], [0.21154245734214783], [-0.2960786819458008], [-0.07350330054759979], [-0.09332668781280518], [-0.3855535089969635], [-0.2193688452243805], [-0.24045804142951965]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_4efc31598561e7d713b8ae04a430366d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.3309921324253082], [-0.14576321840286255], [-0.26363712549209595], [-0.23116803169250488], [-0.16991038620471954], [0.25445711612701416], [0.21916604042053223], [-0.4335433840751648], [-0.18733468651771545]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_8888d0bae9dd0201d5c0728a453e43fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a0e64daab47f18d2ce4978eb638cfb1
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af942cfeec496831e65a813c50c98c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00ffcd55bcef04018aa126132337de44
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.43791815638542175]], [[0.14317414164543152]], [[0.09789053350687027]], [[0.4941061735153198]], [[0.3782956004142761]], [[0.2173541635274887]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_4cde93f28b9212d6c0e2272c312bea30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cde93f28b9212d6c0e2272c312bea30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ee52b249079d606f912eaf9d24a016c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e84c6fd1d15005e3e4e63e2d3e6c693b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7f4aca9fb890defdceddbc5ad1f1b0d
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dae1c772d9348d76189681d09712289f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_838da6bcf54dbd6ac5be09e893c2e00c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dae1c772d9348d76189681d09712289f
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b35f5f86a9a4b9197b9613fb4bca0d14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04cf1499a215f4d8cc7b20e1004e075b
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca15607636da762ab543fd54fe8ca99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca15607636da762ab543fd54fe8ca99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dd146855b6779d875004a9806185978e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = -2.0
            input_2 = 15.989999771118164
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aaa92b846de0c1306ac839f9e3190135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd146855b6779d875004a9806185978e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1673f3f0eb82ed330fde95318ff3ced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7f4aca9fb890defdceddbc5ad1f1b0d
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_19201e9c360cd507e2b79962ba7cfc46(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, 100, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_edb48d52f5b41ac57bd255d834a5d707(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19201e9c360cd507e2b79962ba7cfc46
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e45acdde721213eed6e8f636558e6baf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a0e64daab47f18d2ce4978eb638cfb1
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_abbcd6ca5b7042d0bca00b6c89c06632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7f4aca9fb890defdceddbc5ad1f1b0d
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22a99795aaab38397287a64ef81ee2ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1cfc8524a72d202e95e716d4a37b773
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19b3717680f1255e0e6128499e79f544(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19b3717680f1255e0e6128499e79f544(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73ba2b6f32cc29f8d940ed967e64f3c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ffffd2aed5567636f118a4574005a74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a0e64daab47f18d2ce4978eb638cfb1
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26bd21051ed6666f8b276a4e2a316081(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dae1c772d9348d76189681d09712289f
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_228bdebc2f471f3e4f32cacc8033e65d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07224521040916443]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9173dcdd6b1d5cd5644c34ab57bb4803(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.00883309543132782]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_59c48e96d74d1e8c8589cbaa88d8156e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2857878804206848], [-0.1403680443763733], [-0.09240537881851196], [-0.35986006259918213], [-0.34030163288116455], [-0.09420964121818542]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d041c3f8e06de65291a30c277f41e038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.11981570720672607], [0.14607250690460205], [0.1565159410238266], [-0.11083714663982391], [-0.16156813502311707], [-0.09651578962802887]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_a3a52ccd00fa651ae7eb7fde98e8bdfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1cfc8524a72d202e95e716d4a37b773
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_82eaba1a2fcc887631e047494f85df3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46e56844ead73d0cc941b176514e29ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82eaba1a2fcc887631e047494f85df3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1673f3f0eb82ed330fde95318ff3ced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7f4aca9fb890defdceddbc5ad1f1b0d
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_447355034afbf682b229119a184e868c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_447355034afbf682b229119a184e868c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ee8e1377ddfeaa9d6d87405a3da30f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b23cb7ba032bce01d109bad76fc2027c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b23cb7ba032bce01d109bad76fc2027c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ba6e361134a001e24c58c18e9dd1ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc738b98b2e873b2c066697ad86366c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fc7e45470ecbec44798a02656659898
        def get_inputs(self):
            return [
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_50c2779c480236cfbe2e9580636de1f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50c2779c480236cfbe2e9580636de1f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4be489ea89a78be427bf6bef3a7ab042(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d3c8b4a3d4251c61e26805bcd1560a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.11819621920585632], [-0.2689628303050995], [-0.422619491815567], [0.09488201141357422], [-0.2708263397216797]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_128ed9937807990fcd29f7d8f0a6036f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2624059319496155], [-0.3319106101989746], [-0.3877258896827698], [-0.16239288449287415], [-0.3994396924972534]], dtype='float32').reshape([5, 1]),
            ]


    
    class PrimitiveOp_14137e17edb8c35dadd70549973d41b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1248], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c450c9a3d7a4a7e81d4a454073510041(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14137e17edb8c35dadd70549973d41b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_791b4992de144fe3743591da7a6057b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04cf1499a215f4d8cc7b20e1004e075b
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10976eed36233fc9531ae63ca9da1d9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dae1c772d9348d76189681d09712289f
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96bf2b1a1c379c0b40c3c192bc00f5d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96bf2b1a1c379c0b40c3c192bc00f5d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae22b85d47813a57ad9d5549a5718bb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45cc74a73cdf67ef2564acae3084562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f45cc74a73cdf67ef2564acae3084562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c86f3eae4e8dca90e066ec8f5fa839a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef84a6702f13e93fb7a6325b25ecf596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef84a6702f13e93fb7a6325b25ecf596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddfa2b87b421368322ec4100bf137b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_462d79727e92737b3b319432c3e30b25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 156], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7b204677b689ed7b8115c19660a16c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_462d79727e92737b3b319432c3e30b25
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c136bd5c1dd47bc8549b5aa16aadc9fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2632b854a8aa16888ff566b5f76f56cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab8c06bc4d44fa3c8cd7af51ffc14cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04cf1499a215f4d8cc7b20e1004e075b
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c4f5936dc0bbfbf71eab9c54d8892fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_04cf1499a215f4d8cc7b20e1004e075b
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a87696c9c234c198218356b21ad12a04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dae1c772d9348d76189681d09712289f
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd5ffef621273a3ed49f7d61d4dbf68b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fcb0b662853a3e4c07ba0b707c7f8f71(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd5ffef621273a3ed49f7d61d4dbf68b
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46e56844ead73d0cc941b176514e29ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82eaba1a2fcc887631e047494f85df3a
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f44f79bad051bf9edc5abbf8b19fdc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.24875834584236145], [-0.1563933789730072], [0.009707748889923096], [-0.0416700541973114]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d18d7fceae9eef037da98e294812a001(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.003068208694458008], [-0.19393792748451233], [-0.2989177405834198], [-0.136785626411438]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_2586131888b83e61d82a5db0cbfe0dd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2586131888b83e61d82a5db0cbfe0dd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ee8e1377ddfeaa9d6d87405a3da30f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c05c79d256dd3a00c1dd8f8f21375d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c05c79d256dd3a00c1dd8f8f21375d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_808b63b17fb1b7e688a82dd80037b93b
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_007b2df2fdc09d149d366c9a77a81d7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f4e207db3f09b0e4b52b1006f50e09c9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 624], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c35ffb25c4823c8d2cad767462a7a5f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4e207db3f09b0e4b52b1006f50e09c9
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d74828e4ded7f9e670e3a84c0315001c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39f80e7ecbbff344e89c0850c9d99a81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0ef5bb7823666ca53db2a4edea933fac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe730d82ed4936a8274ba29290a9b259(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ddd2a898f7d3cbbafb7ab4091c1c2d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00ffcd55bcef04018aa126132337de44
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.4248183071613312]], [[0.1123322993516922]], [[0.19716881215572357]], [[0.3563118278980255]], [[0.3588925302028656]], [[0.3417365252971649]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f7d8e698d72638081f7231675fe47d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97f416eb519310fec9e8c4f13244522e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0d075d8b2e69d6c36713c27486df7f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fc7e45470ecbec44798a02656659898
        def get_inputs(self):
            return [
                paddle.to_tensor(8732.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_a731d30b61c73193dedd4e66c271a883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a731d30b61c73193dedd4e66c271a883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([145, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f552e41d170b05e2239223b8ed645453(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([145, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a2f52279d860430d2887a1bf9d026b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 23, 23, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2962711a6e1d5c40b66db3c3bc8ece76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([22, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a82268df19ac55809d3b4a172a59012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44452be7ff8ebac182440798a6cc97e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44452be7ff8ebac182440798a6cc97e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([1827, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4912c94c350068f05fb5a29229270d00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_348cf5311743faf201e6c191c6eeafe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_baeb93708d0e6e194a5bae54cff3268c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2542710304260254], [-0.1396092176437378], [0.21154245734214783], [-0.2960786819458008], [-0.07350330054759979], [-0.09332668781280518], [-0.3855535089969635], [-0.2193688452243805], [-0.24045804142951965]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_4efc31598561e7d713b8ae04a430366d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.3309921324253082], [-0.14576321840286255], [-0.26363712549209595], [-0.23116803169250488], [-0.16991038620471954], [0.25445711612701416], [0.21916604042053223], [-0.4335433840751648], [-0.18733468651771545]], dtype='float32').reshape([9, 1]),
            ]


    class TestPrimitiveOp_6c96673c91327d8fed02417fc1c10357(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af942cfeec496831e65a813c50c98c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00ffcd55bcef04018aa126132337de44
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.43791815638542175]], [[0.14317414164543152]], [[0.09789053350687027]], [[0.4941061735153198]], [[0.3782956004142761]], [[0.2173541635274887]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_d59ad4b31a1d2e52471dfa8d06b88689(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d59ad4b31a1d2e52471dfa8d06b88689(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([5514, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ee52b249079d606f912eaf9d24a016c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f7d8e698d72638081f7231675fe47d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_358bf63927b6fbbe58d88dd7c2c79c45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a37d685f4206cd0ed3afc4478938b4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([10, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1638f796178690b3553ecd67a700ed66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1638f796178690b3553ecd67a700ed66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([1799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaa92b846de0c1306ac839f9e3190135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd146855b6779d875004a9806185978e
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5ae4d737619b1ace83161bfab9ef814(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c83b6afaab9993f2e98423cdc92e8760(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.clip(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9a42aa913ae4f06e3ac1dcf0614897d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c83b6afaab9993f2e98423cdc92e8760
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 100, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f8f171dee9d4f8cd2ac2024f4a9f91ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([171, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_348cf5311743faf201e6c191c6eeafe9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([171, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d50a39b7a54b1c9bf99c663031f0e56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([171, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4d954df2984281078b43be8f443065c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 42, 42, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e61a85ec8ba02d480cfd8f41cd89be85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 21, 21, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_562255c7c185097ee8be2b264163e664(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_562255c7c185097ee8be2b264163e664(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([1503, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73ba2b6f32cc29f8d940ed967e64f3c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1dd27e10c5f02dc112fb8d91bf0ffdc3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 12, 12, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02a90de25f2b205147bccea066ed8360(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([22, 240], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8641ce23f38e368c52a7deb4d0379b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 11, 11, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4961438969ac592bd60e8ad9f7cbdc13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([22, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_228bdebc2f471f3e4f32cacc8033e65d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07224521040916443]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_9173dcdd6b1d5cd5644c34ab57bb4803(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.00883309543132782]], dtype='float32').reshape([1, 1]),
            ]


    class TestPrimitiveOp_59c48e96d74d1e8c8589cbaa88d8156e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2857878804206848], [-0.1403680443763733], [-0.09240537881851196], [-0.35986006259918213], [-0.34030163288116455], [-0.09420964121818542]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_d041c3f8e06de65291a30c277f41e038(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.11981570720672607], [0.14607250690460205], [0.1565159410238266], [-0.11083714663982391], [-0.16156813502311707], [-0.09651578962802887]], dtype='float32').reshape([6, 1]),
            ]


    class TestPrimitiveOp_08c56a5b10f5efa0b060875c58591073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([145, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f416b8108aa3b64022ef4701ed03ce17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5ae4d737619b1ace83161bfab9ef814(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([22, 336], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0a18cb9ebf137cda7caed5cc5634f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 48, 48, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c693d44bfce3cca9676f4315ec0795fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 24, 24, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac1ec4922113aa6d922e9fe5357070af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac1ec4922113aa6d922e9fe5357070af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([2077, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ee8e1377ddfeaa9d6d87405a3da30f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_881666b712d2b0aae07f14a1c5874588(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_881666b712d2b0aae07f14a1c5874588(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([4628, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ba6e361134a001e24c58c18e9dd1ad9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc738b98b2e873b2c066697ad86366c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fc7e45470ecbec44798a02656659898
        def get_inputs(self):
            return [
                paddle.to_tensor(2434.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_b66f036003e51364b3141bdeb4035f8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b66f036003e51364b3141bdeb4035f8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([1101, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4be489ea89a78be427bf6bef3a7ab042(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dcc3c08b617c283a239f0a9c113a8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 46, 46, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2bd983a7d09470970fdea2a1f7d1631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 44, 44, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d3c8b4a3d4251c61e26805bcd1560a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.11819621920585632], [-0.2689628303050995], [-0.422619491815567], [0.09488201141357422], [-0.2708263397216797]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_128ed9937807990fcd29f7d8f0a6036f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2624059319496155], [-0.3319106101989746], [-0.3877258896827698], [-0.16239288449287415], [-0.3994396924972534]], dtype='float32').reshape([5, 1]),
            ]


    class TestPrimitiveOp_bb543e13d4f440c084d7af6a5109b1b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 1248], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94959837c524ee414bffca46a52e99d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 38, 38, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97094ad1888faf7ead072782081386cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([171, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6318b16da543f7e076617a6a2293e784(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 92, 92, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_101b8c62d52c398be95fc0a1a9b9aba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([145, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b84ea797dcdf65db79054ad3a503371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b84ea797dcdf65db79054ad3a503371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([2361, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae22b85d47813a57ad9d5549a5718bb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dd9ab0be26f42032c1fd90e21046950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dd9ab0be26f42032c1fd90e21046950(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([3061, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c86f3eae4e8dca90e066ec8f5fa839a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_254e2f7a36dd352c3148ef7fb48b759e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_254e2f7a36dd352c3148ef7fb48b759e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([3799, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddfa2b87b421368322ec4100bf137b62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4dd6ca943a151de9f0fb346e6efb814(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 156], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46eadb32c5501bf152fe4bd0c54d607f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 22, 22, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a82268df19ac55809d3b4a172a59012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 872], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_656351a59ea2b1b7ee5cf77774a4ef26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([22, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0c4439394258ad60b378fb6ba364e15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([145, 480], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d1eb3500f858ec2c2c26a8cf3324c181(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([171, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f9c5fcfb9281a0bf7faefd6772270fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f416b8108aa3b64022ef4701ed03ce17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 672], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f44f79bad051bf9edc5abbf8b19fdc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.24875834584236145], [-0.1563933789730072], [0.009707748889923096], [-0.0416700541973114]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_d18d7fceae9eef037da98e294812a001(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.003068208694458008], [-0.19393792748451233], [-0.2989177405834198], [-0.136785626411438]], dtype='float32').reshape([4, 1]),
            ]


    class TestPrimitiveOp_f462de50bf7b42d277e14678d9380e65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f462de50bf7b42d277e14678d9380e65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([2088, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ee8e1377ddfeaa9d6d87405a3da30f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57d79d48f8c3e62ac4b36148ed623d5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 19, 19, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6b1abc5bdb64dd837d0eebcbcec8fd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 84, 84, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a85f21cd1f498343788c49719665abe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a85f21cd1f498343788c49719665abe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8bdbcc27c82bdc2d534b7d6537c6a990
        def get_inputs(self):
            return [
                paddle.uniform([4270, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_007b2df2fdc09d149d366c9a77a81d7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9157e9ccc60f8936d57d3c6165c9fe24
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2788627812153024f101b3e2e2cf3261(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e85bbf6a2020d0678a2c3096e52229e8
        def get_inputs(self):
            return [
                paddle.uniform([1, 624], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0493518ac57008e08274ce4f07489a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f4cf19edbfd93aa5805c1595cddaffaa
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 76, 76, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()