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
    class PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, ):
            input_0 = [11, 1, 1, 1]
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    
    class PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, ):
            input_0 = [43, 1, 1, 1]
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    
    class PrimitiveOp_6ec344e14fe8c3e6e7e9375b519d6c5b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, ):
            input_0 = [1, 64, 1, 1]
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43391444d9c1a8c1b66ba252af5b21d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ec344e14fe8c3e6e7e9375b519d6c5b
        def get_inputs(self):
            return [
            ]


    
    class PrimitiveOp_b66bb289961fac3f19550876608049f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, ):
            input_0 = [1, 512, 1, 1]
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    
    class PrimitiveOp_950455c288976297203969de589bdb92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, ):
            input_0 = [1, 192, 1, 1]
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be0703c64b50da3fd3d703ecc7851fb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950455c288976297203969de589bdb92
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_43391444d9c1a8c1b66ba252af5b21d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ec344e14fe8c3e6e7e9375b519d6c5b
        def get_inputs(self):
            return [
            ]


    
    class PrimitiveOp_3b89213f5f93b8f1e818162a3483d9a6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, ):
            input_0 = [1, 256, 1, 1]
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_66cb4e3bb408ae68818f5d9657dc28cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b89213f5f93b8f1e818162a3483d9a6
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_66cb4e3bb408ae68818f5d9657dc28cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b89213f5f93b8f1e818162a3483d9a6
        def get_inputs(self):
            return [
            ]


    
    class PrimitiveOp_97d11a91d6da5cd7e5d71b619af3f2c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, ):
            input_0 = [1, 128, 1, 1]
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c7391e84cbd27f00b1f6b1b2786339d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97d11a91d6da5cd7e5d71b619af3f2c0
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    
    class PrimitiveOp_1e9f12577ae756740b83ada3b75436e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, ):
            input_0 = [1, 2048, 1, 1]
            input_1 = 0.0
            input_2 = 1.0
            return paddle._C_ops.uniform(input_0, paddle.float32, input_1, input_2, 0, paddle.framework._current_expected_place())

        def get_input_spec(self):
            return [
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac392388e77dce02e0ac7b3ba077ec0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e9f12577ae756740b83ada3b75436e7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_ac392388e77dce02e0ac7b3ba077ec0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e9f12577ae756740b83ada3b75436e7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_2c7391e84cbd27f00b1f6b1b2786339d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97d11a91d6da5cd7e5d71b619af3f2c0
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_66cb4e3bb408ae68818f5d9657dc28cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b89213f5f93b8f1e818162a3483d9a6
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_2c7391e84cbd27f00b1f6b1b2786339d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97d11a91d6da5cd7e5d71b619af3f2c0
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_43391444d9c1a8c1b66ba252af5b21d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ec344e14fe8c3e6e7e9375b519d6c5b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_be0703c64b50da3fd3d703ecc7851fb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_950455c288976297203969de589bdb92
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_43391444d9c1a8c1b66ba252af5b21d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ec344e14fe8c3e6e7e9375b519d6c5b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_66cb4e3bb408ae68818f5d9657dc28cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b89213f5f93b8f1e818162a3483d9a6
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_66cb4e3bb408ae68818f5d9657dc28cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b89213f5f93b8f1e818162a3483d9a6
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_2c7391e84cbd27f00b1f6b1b2786339d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97d11a91d6da5cd7e5d71b619af3f2c0
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_ac392388e77dce02e0ac7b3ba077ec0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e9f12577ae756740b83ada3b75436e7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_ac392388e77dce02e0ac7b3ba077ec0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e9f12577ae756740b83ada3b75436e7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_e038641f52a8a447e20d9061ecc1dced(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67e1af2821327ffa7602e7edf74cb03b
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_2c7391e84cbd27f00b1f6b1b2786339d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97d11a91d6da5cd7e5d71b619af3f2c0
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_66cb4e3bb408ae68818f5d9657dc28cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b89213f5f93b8f1e818162a3483d9a6
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_f6c7be68b68799ef4071655481c980a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2167edd4494fb1b0ac5d271e36035bd7
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_d285407037fac184188d9b33bdde6efc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b66bb289961fac3f19550876608049f2
        def get_inputs(self):
            return [
            ]


    class TestPrimitiveOp_2c7391e84cbd27f00b1f6b1b2786339d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97d11a91d6da5cd7e5d71b619af3f2c0
        def get_inputs(self):
            return [
            ]


    

if __name__ == '__main__':
    unittest.main()