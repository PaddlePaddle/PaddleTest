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


    class TestPrimitiveOp_7620e28203836b4530dc360c07780549(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f42889468276bc2a0bacd84c29573c75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2799b6643d801e30bbd4d6613a83083b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_792050cdcbf22d9db069345f6b550d54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5357ab81714a6939c63a9745147e180(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3ccd2db34581fa67c0eb39ab89d87f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_757287129ec3970e3ce790fd8b7b454b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bffd68fd421ec6f497d60bcd69129e84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79f4d81fe2f5c1e27ca17644001c76cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c23be1d8a486786ed48ee8506f795b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0ee12af72dc2f71db9978c1d4148943c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de369a2173c994446f06dacc962a34ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afdbee7cb8fbdb9a347d5e038cabdb3e
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_960b4e7ecb886b310e5b08c520b93454(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53a849d4b16ae468607c67ad3028c7f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_960b4e7ecb886b310e5b08c520b93454
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1cf351c1afd0ae92673586b351253744(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84a80f6cdeb7eb35fe746c959bdbf77c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1cf351c1afd0ae92673586b351253744
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8b80c4bd95b4c2eee6625c462e3b538c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_37f4ae4c9d00941292e32a184a29a3ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8b80c4bd95b4c2eee6625c462e3b538c
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0750d9ee6ab4582b239ff6a14cdef55a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c43900985d60ca0e8a618de4789f3911(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0750d9ee6ab4582b239ff6a14cdef55a
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_529987cf1eb2dbeac1bbad0f24fe4dff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d39d009128356a7547535d11e9b2258(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_529987cf1eb2dbeac1bbad0f24fe4dff
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f2e440cbc988e261a5508f7112a13950(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d08beb3753bc983f121980aedb87d25e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2e440cbc988e261a5508f7112a13950
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_54cf6fe6ef14b7a3a030eee7a4ac6309(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3a1e2c34176f0db72b2cf8fa37c7a902
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ecb2b5392a435c34b61904a13eb49efa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b54b572daab51a7b784d03c167db19c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecb2b5392a435c34b61904a13eb49efa
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_776a9bfcd89b158b59969e7ae835e13d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f12bb0c4b7a2f63fc1116e8a5aa7b814(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_776a9bfcd89b158b59969e7ae835e13d
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5a782650c0257eae677f466094abcdf2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0d24334c9dca0a44e0f4dd7f7333e5ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a782650c0257eae677f466094abcdf2
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_762eeac22e3bee34a1c6fb2e36dd78c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_677e85614f0b80aa7eb96f49b7dd1956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_762eeac22e3bee34a1c6fb2e36dd78c2
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a3dbba1d16f6cc7d7a5723790d1efa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8459f41aee1099fd6a096822de03083
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eabc9ce93d43f2aa9ff84daa152e2e51(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd16a760f4d15ecb006e69bef69aba50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eabc9ce93d43f2aa9ff84daa152e2e51
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_78737c8afcd9834e213959c70b90d196(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.mean_all(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30b4bb0d5da2e171ed651fdc46b37e74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78737c8afcd9834e213959c70b90d196
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b8245c31e1e2157ade36fc05705ab460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77b3f1bc20948130b34bb69925a30186(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_012914b62b6f7746819e9422d746b824(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c0a631a6d4bb07e1ff2e3177e926ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a51d3b05389b39fcee214335c9f6e2cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e59745fb3c58c68fcbc4113fb2f59edd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_444715e70b72ffca8f1be2079093b40b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9ed29b38cff14a8bbff6aa5f2298680
        def get_inputs(self):
            return [
                paddle.uniform([6, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b25ec547435f31038bf22df67b8d7f9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e73664a47cbff6910819a1aea45b1c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b413a74396cf644e2f3d0f09e132d60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5b64adbda8f08bddb279534bce73f0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adef9af867f501ccd614aa1608d8b70c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f9ed29b38cff14a8bbff6aa5f2298680
        def get_inputs(self):
            return [
                paddle.uniform([2, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cb9cbcb06e58c68eaf57bf38042a94b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_348d8a3cb1955181ae385fda341112d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_547c8646b5d9029ff035bf9233a6d017
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()