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


    class TestPrimitiveOp_09e064ffff4c7bbb87bdfc883e8033f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_accfbabca0d1620f42218e7ef6c29c01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c760307d512646eb6eb083cce085d46f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54a1a8534307028d19f9d13bba576317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_649f94c607b16f534855de4f9deec079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e611e527c37048b238fb7ce7b125c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_1ab922c5380c04841a7bd71a11a7416e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a9e7aef7ac7785d55cc5a0eca621751(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee1176ccbdca607351d152231469eb52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_151fc8aa1995be894e59aa3e8d7ae58a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e853aa9bc64c20f6fcce27cc348f686b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee7899f45c681bce34c079c089040db7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_94cb3606aa20fd6e625fbf8ec3d9cf35(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f3f9c0e2f2a31bd39622cec7fcef2fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94cb3606aa20fd6e625fbf8ec3d9cf35
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e306d96c785aade82d558cc260c2f77a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ad0bb5f2ca65743fcb461e443fb43ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e306d96c785aade82d558cc260c2f77a
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2068f7fd22747ef656b51dc22f4b6230(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_92bd6d664b55da78001b2724f4d79e1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2068f7fd22747ef656b51dc22f4b6230
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_411b222022a27aeec6e3392ad2c39324(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63fdfdec4b2e2688f8a83aa281f3e9b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_411b222022a27aeec6e3392ad2c39324
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a43f8c39048dc331366fdfd372ee93e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4d72404dbe1bd77e8429430737aec7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a43f8c39048dc331366fdfd372ee93e3
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fd78c600b1f2669c6be4cfac904e9308(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a90edd0206e43ab7350e720078d809da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd78c600b1f2669c6be4cfac904e9308
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54cf6fe6ef14b7a3a030eee7a4ac6309(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a1e2c34176f0db72b2cf8fa37c7a902
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_846f6b827cfafdad87610fc28bc4eb88(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e697d56457a805148fe4d039cf41af4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_846f6b827cfafdad87610fc28bc4eb88
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_13541eec6d0185708e319761922c36e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c10fcb3cd7507f0b1654006734ab47e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13541eec6d0185708e319761922c36e3
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df3a1c86d3a5f0334608c156e51671bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2fb751bd701b8d7c12b421e5a137e065(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df3a1c86d3a5f0334608c156e51671bf
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a933ae0c6e92912fe58e3b9b2e8b0e9f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5926d61a24260b441b1dcd262932c12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a933ae0c6e92912fe58e3b9b2e8b0e9f
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a3dbba1d16f6cc7d7a5723790d1efa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8459f41aee1099fd6a096822de03083
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1483dc7e6f79dd0df490cd384ae90cb7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd6ca24be67904a324cf8a607d1f86b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1483dc7e6f79dd0df490cd384ae90cb7
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d45fc9bc958705b95245cf3de69f0633(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8992272fae1e24f3ea599d105df400e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d45fc9bc958705b95245cf3de69f0633
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4ea67884613b30111648997cc11f546f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7feef35e8ec0113cd527144ba39c610(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4eeb8edcfb80fff982a930ee0c4e831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0535b366dcfa4f52a3db13024da4990(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8310bf9967d83af63aa3af89e045674e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d3e1b94ea9726837ffbe5b478e3a886(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_444715e70b72ffca8f1be2079093b40b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9ed29b38cff14a8bbff6aa5f2298680
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_137f989400163c0065470b52c27f6b14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aeee947c64b4dab974cce6e93cdd1338(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bb13444d1de6a9449bc9bf07354e73f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15a9a365a8de641a02a26f6b0b5cef21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adef9af867f501ccd614aa1608d8b70c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9ed29b38cff14a8bbff6aa5f2298680
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ffaefdcc70919d627b84ffc6d795603f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfa613304792651b7e4cfb336353dac5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()