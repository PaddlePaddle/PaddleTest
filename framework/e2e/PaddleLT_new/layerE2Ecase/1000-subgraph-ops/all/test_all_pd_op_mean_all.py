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
    class PrimitiveOp_29a0706b2ad0744e2413ae3ec2416b48(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5731195258b16ae6cb0c49f7fa5e7bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29a0706b2ad0744e2413ae3ec2416b48
        def get_inputs(self):
            return [
                paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f75d24873073c60a11e6a53ef27b9b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_add0436d85952795f17e9ca376c1cce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f75d24873073c60a11e6a53ef27b9b5
        def get_inputs(self):
            return [
                paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7d4bbf66e8ff0de849e27680b8b5149a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5471d3b5e4b279cd54ea54e5dc5bca94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aabd55f05565676016c1149adb6e2944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7d29eedf4fd9afc25851f91da275af8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f232b7acdcc8a62f660d98515ad0d2fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea32cad8ff57ea09ca9a41ca35f0c7ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3a1e2c34176f0db72b2cf8fa37c7a902(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_54cf6fe6ef14b7a3a030eee7a4ac6309(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a1e2c34176f0db72b2cf8fa37c7a902
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf4cc53b9cec6709c57c57fa57c5e3b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16ad0b75b7e3f0b9cfe4b83a3d0037dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73f6387a1a89fa5b32198a23d5de10ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b84861045a12af83d82bc720bb9805b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a8459f41aee1099fd6a096822de03083(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a3dbba1d16f6cc7d7a5723790d1efa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8459f41aee1099fd6a096822de03083
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c0cf6123376871a5a6ce8eb0a38047e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01a2033209087b05caffd46d23afb43c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5731195258b16ae6cb0c49f7fa5e7bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_29a0706b2ad0744e2413ae3ec2416b48
        def get_inputs(self):
            return [
                paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_add0436d85952795f17e9ca376c1cce1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f75d24873073c60a11e6a53ef27b9b5
        def get_inputs(self):
            return [
                paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9681658bf6548ebb12b250ca57c1560e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46bc1c8947385e1e46780a20920cb4ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9681658bf6548ebb12b250ca57c1560e
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c598f2c59cda6e84d616edd21c197eb8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b27b454b2244f21b311f86f4091ac77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c598f2c59cda6e84d616edd21c197eb8
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6eb21352cba32cb59d1478efd4ab9cca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0ca0e1d343745c825d2207226e16cd06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6eb21352cba32cb59d1478efd4ab9cca
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0d78368cae154646b504aa5067f21a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0103e9136c2531efccf3f1e0b03733b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0d78368cae154646b504aa5067f21a5
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a7e86a3258922a1895de01dc697c042d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee589308992be54d755b9bc39c65a48c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7e86a3258922a1895de01dc697c042d
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cdf217fe7225a3bde6221c362b893d33(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0804a32bdf7b3650f811e01ee819fe25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdf217fe7225a3bde6221c362b893d33
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54cf6fe6ef14b7a3a030eee7a4ac6309(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a1e2c34176f0db72b2cf8fa37c7a902
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bcc0d8021169acdedb66a22eab8a9fc5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27240c9c684932b42e8605c1a84ef673(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bcc0d8021169acdedb66a22eab8a9fc5
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a5042bcc88dffbad316973ed1042fa05(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_88dde92608b6f6bad4363af56873c15c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a5042bcc88dffbad316973ed1042fa05
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f797f111b722e86940a5d9f6441ce70b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12e7b80d841ac051497eeacbec4506d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f797f111b722e86940a5d9f6441ce70b
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e48f20482abe674540a1fb2eaff2dc5d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_237191e70aa55f7d22f9c02a9c107dd8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e48f20482abe674540a1fb2eaff2dc5d
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a3dbba1d16f6cc7d7a5723790d1efa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8459f41aee1099fd6a096822de03083
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c03d66a90fcb68049579e5a04e3d2ec8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37e867191cfd655d0868fec6b6a4f918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c03d66a90fcb68049579e5a04e3d2ec8
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0f577d3719b9fffcf2a49bf01a017867(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_df06c32a0508638f084881f2c08fb066(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f577d3719b9fffcf2a49bf01a017867
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f9ed29b38cff14a8bbff6aa5f2298680(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0c22c152497cafdeeaeba39d85e2ee63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9ed29b38cff14a8bbff6aa5f2298680
        def get_inputs(self):
            return [
                paddle.uniform([4, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b565b312e798045c5a32652379cd2f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9ed29b38cff14a8bbff6aa5f2298680
        def get_inputs(self):
            return [
                paddle.uniform([3, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_547c8646b5d9029ff035bf9233a6d017(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1b893e43aaeaae443197f57e504af55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae8c5ec59a11fff44f1d5853a7be7131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb0fe98441cad84161a220646dbd830f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9bfcffbf0a7b7558ec689d09e55a6d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6d92380b7a447547c25a897d59943fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_329ae24ee883befa7e9f09ec742adfec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_444715e70b72ffca8f1be2079093b40b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9ed29b38cff14a8bbff6aa5f2298680
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ba337027cbfeffe92fbcd6b5234adc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8b6bd6f1ce3e1ee7cfa19defa01163a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb32f25d9f16cf40ee9739e91c1d4f1f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53cb96ad2d3ebc40b1a19390ff62e934(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adef9af867f501ccd614aa1608d8b70c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9ed29b38cff14a8bbff6aa5f2298680
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4ce0945bc24e5ced69011facbab1482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89e7511e9676a65c3a2172bcbe7d8293(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()