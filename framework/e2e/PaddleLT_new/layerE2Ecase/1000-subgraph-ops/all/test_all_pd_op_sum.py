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
    class PrimitiveOp_77143eb31096c3543f9d60ee35bc152d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 2, 16, 9, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38b4b3dbbfcf6c3867dbbfdde2226c3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77143eb31096c3543f9d60ee35bc152d
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[0], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e011ab6de85328141e6a17d126a65852(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([4354], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_74ccd3eb6a0db5fa2d98f4d61afccede(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[0], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0860a4a21695f5da810b1d51cf066660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ccd3eb6a0db5fa2d98f4d61afccede
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[0], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7708bdf174d8e23bcb07effd2ffd4c3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02e7604769688f9a36bf2624176fd032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_02e7604769688f9a36bf2624176fd032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3adb0e0e40abea21c5f7320453dee4fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.0029696375131607056, 0.008893460966646671]], [[9.824388689594343e-05, 3.097688386333175e-05]], [[0.041273776441812515, 0.03675917908549309]], [[0.0012482624733820558, 0.044441789388656616]], [[0.03526579588651657, 0.1963946372270584]], [[0.08096426725387573, 0.09441264718770981]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_22d4d49ae1e2c152f672069724559870(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.09834550321102142, 0.06611254066228867]], [[0.03077700175344944, 0.05494692549109459]], [[0.02604793757200241, 0.03245082497596741]], [[0.017015304416418076, 0.004119689576327801]], [[0.0008794770110398531, 0.013181345537304878]], [[0.014345181174576283, 0.04793524742126465]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_94e7ec325fbf4935be599f369c609aaa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4, 16, 49, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6246cfdf1c32ed9545c56b6526636f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94e7ec325fbf4935be599f369c609aaa
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6db0e0bd6cece53c3749e1564c62d20d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2279099076986313, 0.007394756190478802, 0.1952025145292282, 0.08338708430528641, 0.12485320121049881, 0.07469327002763748, 0.1628420203924179, 0.09415122121572495, 0.195261150598526, 0.008562020026147366, 0.22482548654079437, 0.20597805082798004, 0.009456115774810314, 0.18998557329177856, 0.06557527929544449, 0.017191434279084206], dtype='float32').reshape([16]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8a1a23bacf58b4ddfadacce567417cbc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b0510b072b1d181a080cbe448fdd9c52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7ac05b548a9c4c551a0f54a56eabf274(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de504388c40675b6c69a43e621757b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_895fd06472fffb6cb451bb012c824929(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[0], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea02fbc88bff26269d022b52f20458ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_ea02fbc88bff26269d022b52f20458ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8ba798e1feca437dae341b8a92b76da6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_99f112d5c9f8de9231c24e63d5e230cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.059835709631443024, 0.2431793510913849, 0.06668603420257568, 0.09330688416957855], [0.030676662921905518, 0.4153245687484741, 0.09938794374465942, 0.10088853538036346], [0.041919320821762085, 0.11580070853233337, 0.13541805744171143, 0.10899001359939575], [0.19998770952224731, 0.3514717221260071, 0.011652082204818726, 0.3247314393520355], [0.10408538579940796, 0.18593794107437134, 0.10364016890525818, 0.07083243876695633]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_647d6425039bd307ab167eaf4c7acecb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 4, 16, 49, 56, 56], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d70b478c5a61d918429c4a8477db7a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_647d6425039bd307ab167eaf4c7acecb
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d0be897f5c8030dc9493523d2435948c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07945215702056885, 0.19198602437973022, 0.09651744365692139, 0.051396384835243225], [0.1376815140247345, 0.03616875410079956, 0.030297398567199707, 0.08579754829406738], [0.06492412090301514, 0.02363431453704834, 0.03141921013593674, 0.15331938862800598], [0.1376815140247345, 0.03616875410079956, 0.030297398567199707, 0.08579754829406738], [0.06492412090301514, 0.02363431453704834, 0.03141921013593674, 0.15331938862800598]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5173a1deb9477c53fa1e809e77a221d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ccd3eb6a0db5fa2d98f4d61afccede
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_89ea218c93eebc8ffbc15cbcebbf2310(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9eb622a7a54c03bb2dbd1cef460ca21e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9eb622a7a54c03bb2dbd1cef460ca21e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_2ada1fc8891c860af04a76a4b9bb378c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10889467597007751, 0.25807124376296997, 0.14413365721702576, 0.4194449782371521], [0.3009010851383209, 0.18025171756744385, 0.11982408165931702, 0.061148613691329956], [0.36769217252731323, 0.1304820477962494, 0.4372129440307617, 0.006102010607719421], [0.3009010851383209, 0.18025171756744385, 0.11982408165931702, 0.061148613691329956], [0.36769217252731323, 0.1304820477962494, 0.4372129440307617, 0.006102010607719421], [0.1060771495103836, 0.28093963861465454, 0.11857184767723083, 0.316922664642334], [0.1060771495103836, 0.28093963861465454, 0.11857184767723083, 0.316922664642334]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dd1bd9c3a912665d2ed5f02b964dac28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2a72ca8ab6367501ee4c932512d33193(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 32, 16, 49, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_53ddadc6c0191e0021b057a229cf28b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a72ca8ab6367501ee4c932512d33193
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5adc3daac58cc726e17e9f22a676319e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_de504388c40675b6c69a43e621757b02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_86e4cafb3cc865d16d35e43ff1474986(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_86e4cafb3cc865d16d35e43ff1474986(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8ba798e1feca437dae341b8a92b76da6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_be8e5d4585f72a06044b5897e57221b1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 8, 16, 49, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_446b0e537bd63e5d21a4b379e0ef6d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8e5d4585f72a06044b5897e57221b1
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0d10c259c7b4580fa873f83ddcbaf963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1698136031627655, 0.1802951991558075, 0.08069563657045364, 0.16915643215179443, 0.21542100608348846, 0.16946832835674286, 0.023178303614258766, 0.1980338990688324, 0.018291577696800232, 0.10876372456550598, 0.07114827632904053, 0.08529523015022278, 0.12002154439687729, 0.2009904384613037, 0.17093011736869812, 0.2671310007572174, 0.18313130736351013, 0.20361387729644775, 0.12102005630731583, 0.09615907818078995, 0.24224619567394257, 0.2328210026025772, 0.2187272012233734, 0.12140653282403946], dtype='float32').reshape([24]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d4a89d31c5083d8ac74c6eedd3492cd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e275eb0bfc25d5777537ab5a840baaab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e275eb0bfc25d5777537ab5a840baaab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 16, 16, 49, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ad5c687cd251fa5695c64cb452a70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bcfcde8d3d3d2f05c0e75005a854315f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.03049149177968502, 0.016200125217437744, 0.21287184953689575, 0.11618273705244064], dtype='float32').reshape([4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7503aeee8c6acf044afd919aa0b3993a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22957313060760498, 0.17837467789649963, 0.016854897141456604, 0.40951916575431824], [0.030538499355316162, 0.026057451963424683, 0.00427737832069397, 0.09727845340967178], [0.25486868619918823, 0.3133251667022705, 0.23783250153064728, 0.25276681780815125], [0.15342719852924347, 0.05401526391506195, 0.19605214893817902, 0.08178121596574783], [0.15342719852924347, 0.05401526391506195, 0.19605214893817902, 0.08178121596574783], [0.25486868619918823, 0.3133251667022705, 0.23783250153064728, 0.25276681780815125]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_53f8ea7db0102a4c6d7a8f042a156a0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21438781917095184, 0.07429340481758118, 0.2580089867115021, 0.2164541482925415], [0.13806931674480438, 0.42606988549232483, 0.00037895888090133667, 0.46145597100257874], [0.12322920560836792, 0.12052872776985168, 0.28811296820640564, 0.20321156084537506], [0.05649891495704651, 0.06143069267272949, 0.13873469829559326, 0.31833985447883606], [0.21438781917095184, 0.07429340481758118, 0.2580089867115021, 0.2164541482925415]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_08bd2e2b1823be913ac8378609c293cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9ad5c687cd251fa5695c64cb452a70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_61c86e5ea464bd1b99661ce01b6bc728(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04451256990432739, 0.19473513960838318, 0.36136090755462646, 0.024676114320755005], [0.04359281063079834, 0.4449837803840637, 0.11287054419517517, 0.006923645734786987], [0.2663431763648987, 0.1137508675456047, 0.3312119245529175, 0.14764076471328735], [0.16015861928462982, 0.13828155398368835, 0.009508371353149414, 0.2052965611219406]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_c9be82b37a779006e5c306132eff72f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.sum(input_0, input_1, None, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10ec9a6b58439ff34862782ce1433301(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9be82b37a779006e5c306132eff72f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_446b0e537bd63e5d21a4b379e0ef6d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8e5d4585f72a06044b5897e57221b1
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7695efa726dd5344f89aa265298da191(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_144dc7a4bda71d46fc9fe902db191fcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d125137b97e798a79e92443531e65427(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_606352e16f3ce8adfe8b2381b6c4f503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9d1ecd4c739f58da8e2450ee7e4c9ca6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9d1ecd4c739f58da8e2450ee7e4c9ca6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5fd6e0d823c8fc1913797c338898abf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9be82b37a779006e5c306132eff72f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b4d2459a3bae9673d4eed43d9f9c88b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03216975927352905, 0.04337479919195175, 0.01657208800315857, 0.09818089008331299], [0.03216975927352905, 0.04337479919195175, 0.01657208800315857, 0.09818089008331299], [0.15343743562698364, 0.3510548174381256, 0.23534007370471954, 0.0949157178401947], [0.12012717127799988, 0.08196210116147995, 0.07514175772666931, 0.4337972104549408], [0.061709702014923096, 0.21900923550128937, 0.12456387281417847, 0.1701676845550537], [0.083442822098732, 0.27063196897506714, 0.05398473143577576, 0.12532399594783783], [0.09118098020553589, 0.2385154515504837, 0.08096741139888763, 0.00832115113735199]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_172f6c4046f7a21ca652f8a8283ce6d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5ba56683c1e2d89b2f4e1ad97e466f29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5ba56683c1e2d89b2f4e1ad97e466f29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_45b66f1a722535e9fb77d62c6c205bee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([4849], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_275a4e855928192686f59cbb53d236ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([1206], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_39b290f3aa6c2d170c89ef544d67dc54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_74ccd3eb6a0db5fa2d98f4d61afccede
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_98e1501b3613a136fcdf0278221ef8a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_db01609a1b76497e29164405ec8352c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_db01609a1b76497e29164405ec8352c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f07828cf47b38cfffceb7fe3f356f613(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17413032054901123, 0.02193164825439453, 0.06767009198665619, 0.10391233116388321], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [0.08004461228847504, 0.23684172332286835, 0.0887954980134964, 0.17910811305046082], [0.05183491110801697, 0.12347441911697388, 0.13930287957191467, 0.2521556615829468], [0.21063363552093506, 0.13145971298217773, 0.15847629308700562, 0.02333889901638031]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_f905dd2d33a202374bef7569e6dc16fc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2ea2c79d4be4e31ef9feaac6ecf79fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f905dd2d33a202374bef7569e6dc16fc
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 32, 16, 49, 7, 7], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59f7409ec9197fbd0e416fbeaa8e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_db01bc48173ed80f6896d765edee63fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3fd96cccb5ac8db5709402148075c355(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db01bc48173ed80f6896d765edee63fd
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a2e12a6737afded41b61b97218625263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_04c29b85d787e719d2a0302cb79cbd34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_04c29b85d787e719d2a0302cb79cbd34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_67dd4f2bc5596c6a7a0a01ce54234ff2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_8e9c845111aba231c78b5140f18ecdc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8e9c845111aba231c78b5140f18ecdc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3e95044af55f12ab317b6b0e85e69184(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_2e045feccc21a085d678519d1857d263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_2e045feccc21a085d678519d1857d263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4396c76287f2f41fa85b765be3e7db50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_16344afd53d26fc87d987706cbf46fa7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 8, 16, 49, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ce4730cb65ba5f15186486e234a1e13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16344afd53d26fc87d987706cbf46fa7
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 16, 49, 14, 14], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf88ce40921ac42d3e835d568e89b99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_53ddadc6c0191e0021b057a229cf28b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a72ca8ab6367501ee4c932512d33193
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c13ec1b4872d14ff9e310a01d6fd3d70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.17320916056632996, 0.08482294529676437, 0.09949600696563721, 0.19735485315322876, 0.2445642054080963, 0.24156497418880463, 0.20749948918819427, 0.0047854408621788025, 0.08320922404527664, 0.196404829621315, 0.07307159155607224, 0.05009898915886879, 0.09655003994703293, 0.08765567094087601, 0.11944684386253357, 0.21975986659526825, 0.08828285336494446, 0.10039164870977402, 0.2237415462732315, 0.07407843321561813], dtype='float32').reshape([20]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_62b6792c6743cbb393502eaf575d9b65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([17369], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4ce4730cb65ba5f15186486e234a1e13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16344afd53d26fc87d987706cbf46fa7
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_aa61e97e8af5f9b53cd5313507663df4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_00baa7162cb1acd818deaf6a54f183d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5cb553998a8af8a2cc48cd47a6f32c60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_05334d95faf5d3a41478e1ca6e2edd79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_05334d95faf5d3a41478e1ca6e2edd79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_c9c932fce0d50cfbeda7ffc86a5e04dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 2, 16, 9, 112, 112], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_39016c3c718b96e0ac175b2111e69f48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9c932fce0d50cfbeda7ffc86a5e04dc
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0db75a1e8834dac9c5528ee82c493da2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_ce6188ce7c0724efdad00bdab9c3bddc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36759066581726074, 0.1356358379125595, 0.07934285700321198, 0.07276766747236252], [0.37631040811538696, 0.03991866111755371, 0.0015145167708396912, 0.07648283243179321], [0.07664056122303009, 0.06367561221122742, 0.027359262108802795, 0.13421794772148132], [0.07664056122303009, 0.06367561221122742, 0.027359262108802795, 0.13421794772148132], [0.2624303996562958, 0.038322463631629944, 0.20256070792675018, 0.42651233077049255]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_59f7409ec9197fbd0e416fbeaa8e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_460ff43be3a1414b62203003ade6e987(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_2e709bb2ea6ad6308bb18b8c331f580b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7082d29ca054cce66b3806e9a64b9659(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_20aee50bafc74873e7915222fbf50be6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28fea1593e92e8c5f0e7b4c9fc2febc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3d05191f0818cddee35953466ce9f7c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3d05191f0818cddee35953466ce9f7c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_804b8e09b8a847cc7c95d73312cff639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_ad4b68aa9c4ff5d5259633681012a747(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15342693030834198, 0.16269968450069427, 0.16442686319351196, 0.3595154881477356], [0.0890268087387085, 0.00014069676399230957, 0.25438398122787476, 0.21876737475395203], [0.28361520171165466, 0.2432878464460373, 0.06699240207672119, 0.010794661939144135], [0.15342693030834198, 0.16269968450069427, 0.16442686319351196, 0.3595154881477356], [0.3432466685771942, 0.06492901593446732, 0.25622114539146423, 0.2037186324596405], [0.3985275328159332, 0.0876234769821167, 0.07240909337997437, 0.14636847376823425], [0.3432466685771942, 0.06492901593446732, 0.25622114539146423, 0.2037186324596405]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_aa1454c72de0b09199d854f0254e29b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bf88ce40921ac42d3e835d568e89b99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_38b4b3dbbfcf6c3867dbbfdde2226c3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_77143eb31096c3543f9d60ee35bc152d
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_ce832cab3d1ef762b7a7e059e1a801be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4354], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3fee1db55a6e82893beacddd9c3bb0c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce832cab3d1ef762b7a7e059e1a801be
        def get_inputs(self):
            return [
                paddle.uniform([4354], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_80d7ea76df7cab71b8f62eed7f2fccc4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8732, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c7537c668a7c7a5617668e590592f175(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80d7ea76df7cab71b8f62eed7f2fccc4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2a765369f8effeb5abb83d2646745173(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e807c2127db5201fc65fc3c8aecfdfa3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a765369f8effeb5abb83d2646745173
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_23f6b988ad50ceb8714def96a28d3964(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 21824, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a49d146228f2c1702eb94cfac0dad838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23f6b988ad50ceb8714def96a28d3964
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a49d146228f2c1702eb94cfac0dad838(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_23f6b988ad50ceb8714def96a28d3964
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_01bba0a914ebd1f491ac9c5013dccca5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 1, 2], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60104a143325adb5923078abe52116e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01bba0a914ebd1f491ac9c5013dccca5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.0029696375131607056, 0.008893460966646671]], [[9.824388689594343e-05, 3.097688386333175e-05]], [[0.041273776441812515, 0.03675917908549309]], [[0.0012482624733820558, 0.044441789388656616]], [[0.03526579588651657, 0.1963946372270584]], [[0.08096426725387573, 0.09441264718770981]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b858a70093bd365a22c3b1ca2f881138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01bba0a914ebd1f491ac9c5013dccca5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.09834550321102142, 0.06611254066228867]], [[0.03077700175344944, 0.05494692549109459]], [[0.02604793757200241, 0.03245082497596741]], [[0.017015304416418076, 0.004119689576327801]], [[0.0008794770110398531, 0.013181345537304878]], [[0.014345181174576283, 0.04793524742126465]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6246cfdf1c32ed9545c56b6526636f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_94e7ec325fbf4935be599f369c609aaa
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_1fc6813c50b3e0aa20e38d15ae3e4b43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_59a01f561fd5826d8a178172538f52a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fc6813c50b3e0aa20e38d15ae3e4b43
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2279099076986313, 0.007394756190478802, 0.1952025145292282, 0.08338708430528641, 0.12485320121049881, 0.07469327002763748, 0.1628420203924179, 0.09415122121572495, 0.195261150598526, 0.008562020026147366, 0.22482548654079437, 0.20597805082798004, 0.009456115774810314, 0.18998557329177856, 0.06557527929544449, 0.017191434279084206], dtype='float32').reshape([16]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2501c76fc228dbc4074f7954b6e23efa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d967ac90ae8e3c6bde485844427c9b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2501c76fc228dbc4074f7954b6e23efa
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_a29f4c1bab50ac88bc0644a2fa738f4a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[150], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b583c472e943ab5ff4c3d87484a50337(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a29f4c1bab50ac88bc0644a2fa738f4a
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2b293ee699f31fb94ec74aff25ed9715(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f580956c432b416aabd035fe6cd5a6cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b293ee699f31fb94ec74aff25ed9715
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_b8e73f18898db6b531f2a02a1d366e03(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4bd78313895267b946031817d4a0c1c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8e73f18898db6b531f2a02a1d366e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_46525f0d5aa389cf247efa02152aac8d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d107cd0b984770001daa483a2663791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46525f0d5aa389cf247efa02152aac8d
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8d107cd0b984770001daa483a2663791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46525f0d5aa389cf247efa02152aac8d
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_560e12b283db9a4883f57e5f27721a38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[15200], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2050603a276a2955f485b5d686803ea1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560e12b283db9a4883f57e5f27721a38
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_eb83881230c2be7127771fc7af7c350a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_97add6e51d4bf4380d2cee9ce42c387d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.059835709631443024, 0.2431793510913849, 0.06668603420257568, 0.09330688416957855], [0.030676662921905518, 0.4153245687484741, 0.09938794374465942, 0.10088853538036346], [0.041919320821762085, 0.11580070853233337, 0.13541805744171143, 0.10899001359939575], [0.19998770952224731, 0.3514717221260071, 0.011652082204818726, 0.3247314393520355], [0.10408538579940796, 0.18593794107437134, 0.10364016890525818, 0.07083243876695633]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3d70b478c5a61d918429c4a8477db7a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_647d6425039bd307ab167eaf4c7acecb
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_16bc1386c43b70c534de4cbccadeddda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07945215702056885, 0.19198602437973022, 0.09651744365692139, 0.051396384835243225], [0.1376815140247345, 0.03616875410079956, 0.030297398567199707, 0.08579754829406738], [0.06492412090301514, 0.02363431453704834, 0.03141921013593674, 0.15331938862800598], [0.1376815140247345, 0.03616875410079956, 0.030297398567199707, 0.08579754829406738], [0.06492412090301514, 0.02363431453704834, 0.03141921013593674, 0.15331938862800598]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_b39b3d49a514b6e7139b8c486993bfe6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21824, 15], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e174e64c8e2ee3a78ee086b6aa0e4430(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b39b3d49a514b6e7139b8c486993bfe6
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_0cff9e87f018fadb9e594fea702136d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9e99eef50a533c6dd5a257410cfcd713(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cff9e87f018fadb9e594fea702136d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4b0c3b939281026220a0b2326537ad29(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_401df690899bfa484d1ecf33aaa1a304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b0c3b939281026220a0b2326537ad29
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_401df690899bfa484d1ecf33aaa1a304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b0c3b939281026220a0b2326537ad29
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1720dbcfb8b6bf8fa7ba2486f89887f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10889467597007751, 0.25807124376296997, 0.14413365721702576, 0.4194449782371521], [0.3009010851383209, 0.18025171756744385, 0.11982408165931702, 0.061148613691329956], [0.36769217252731323, 0.1304820477962494, 0.4372129440307617, 0.006102010607719421], [0.3009010851383209, 0.18025171756744385, 0.11982408165931702, 0.061148613691329956], [0.36769217252731323, 0.1304820477962494, 0.4372129440307617, 0.006102010607719421], [0.1060771495103836, 0.28093963861465454, 0.11857184767723083, 0.316922664642334], [0.1060771495103836, 0.28093963861465454, 0.11857184767723083, 0.316922664642334]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_79e69cd7ea52f2675476d949b4ae2007(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_143effb5d3314b58c5de56df971220e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_79e69cd7ea52f2675476d949b4ae2007
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_53ddadc6c0191e0021b057a229cf28b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a72ca8ab6367501ee4c932512d33193
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_2cecba88ecc5ad045b13c158bbc3f8c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5fb414615430dd77f905cd5307b4d435(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2cecba88ecc5ad045b13c158bbc3f8c3
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4bd78313895267b946031817d4a0c1c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8e73f18898db6b531f2a02a1d366e03
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_db14b3c776a8e082875c18ba4a0192a2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da0397f2f981193dcd2fc4f3c2966f6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db14b3c776a8e082875c18ba4a0192a2
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_da0397f2f981193dcd2fc4f3c2966f6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db14b3c776a8e082875c18ba4a0192a2
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_2050603a276a2955f485b5d686803ea1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560e12b283db9a4883f57e5f27721a38
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_446b0e537bd63e5d21a4b379e0ef6d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8e5d4585f72a06044b5897e57221b1
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_abf701896c1b5c2cda506d95ebfa0da8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b14715aed8a358c7b23021ce4a297afa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abf701896c1b5c2cda506d95ebfa0da8
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1698136031627655, 0.1802951991558075, 0.08069563657045364, 0.16915643215179443, 0.21542100608348846, 0.16946832835674286, 0.023178303614258766, 0.1980338990688324, 0.018291577696800232, 0.10876372456550598, 0.07114827632904053, 0.08529523015022278, 0.12002154439687729, 0.2009904384613037, 0.17093011736869812, 0.2671310007572174, 0.18313130736351013, 0.20361387729644775, 0.12102005630731583, 0.09615907818078995, 0.24224619567394257, 0.2328210026025772, 0.2187272012233734, 0.12140653282403946], dtype='float32').reshape([24]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_9e3acbc65041fbb551e7cf50223bcb78(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ead18cb5cb40b1978643afa8f1ef05bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9e3acbc65041fbb551e7cf50223bcb78
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_b5e0762e7cf90681849f137f6d228127(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d41f61c44c021bed14f734942eb29a0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5e0762e7cf90681849f137f6d228127
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d41f61c44c021bed14f734942eb29a0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5e0762e7cf90681849f137f6d228127
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9ad5c687cd251fa5695c64cb452a70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_fc26a21b27eba3cdde678c037676765a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d79b12137297fc59e1335bfa0a11a571(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc26a21b27eba3cdde678c037676765a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.03049149177968502, 0.016200125217437744, 0.21287184953689575, 0.11618273705244064], dtype='float32').reshape([4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_5a73f36534c5f3febd704a5d55d75107(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6122c6eb038410a0c7acadf679a3393d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a73f36534c5f3febd704a5d55d75107
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22957313060760498, 0.17837467789649963, 0.016854897141456604, 0.40951916575431824], [0.030538499355316162, 0.026057451963424683, 0.00427737832069397, 0.09727845340967178], [0.25486868619918823, 0.3133251667022705, 0.23783250153064728, 0.25276681780815125], [0.15342719852924347, 0.05401526391506195, 0.19605214893817902, 0.08178121596574783], [0.15342719852924347, 0.05401526391506195, 0.19605214893817902, 0.08178121596574783], [0.25486868619918823, 0.3133251667022705, 0.23783250153064728, 0.25276681780815125]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_007b8d7b6f97cf63737b79da9fbf054b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21438781917095184, 0.07429340481758118, 0.2580089867115021, 0.2164541482925415], [0.13806931674480438, 0.42606988549232483, 0.00037895888090133667, 0.46145597100257874], [0.12322920560836792, 0.12052872776985168, 0.28811296820640564, 0.20321156084537506], [0.05649891495704651, 0.06143069267272949, 0.13873469829559326, 0.31833985447883606], [0.21438781917095184, 0.07429340481758118, 0.2580089867115021, 0.2164541482925415]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_fbdc79e521f488ee17c21203b9a730dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5be42f93d2b5e6dddb66476d60b3738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbdc79e521f488ee17c21203b9a730dd
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9ad5c687cd251fa5695c64cb452a70a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ead94dd5ba444f6bb871b9c5e348cd5f
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_d4b14492b5693ed40c5ffef38cd64f08(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7cdae1e5159c847ffe7454d8d6f648d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4b14492b5693ed40c5ffef38cd64f08
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04451256990432739, 0.19473513960838318, 0.36136090755462646, 0.024676114320755005], [0.04359281063079834, 0.4449837803840637, 0.11287054419517517, 0.006923645734786987], [0.2663431763648987, 0.1137508675456047, 0.3312119245529175, 0.14764076471328735], [0.16015861928462982, 0.13828155398368835, 0.009508371353149414, 0.2052965611219406]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_80ed203e1f4ee446d2eab177758f4979(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.sum(input_0, input_1, None, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 256, 19, 34], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e871d4faff80bcea71bef440a067511(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80ed203e1f4ee446d2eab177758f4979
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_446b0e537bd63e5d21a4b379e0ef6d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be8e5d4585f72a06044b5897e57221b1
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_99b88328100de8299c75292e48456c84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea8445d8f11ace50dec1eb6f5a9e9bab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99b88328100de8299c75292e48456c84
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_130cd2fde7cdef298e7f6dda389e421e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[950], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3aef3d2205d2c18cb8993f835bab13a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_130cd2fde7cdef298e7f6dda389e421e
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_d50b6a56e0c6d4ddaf39e73e80c9c908(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8816], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e27627733f91d33f053f74a33937c273(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d50b6a56e0c6d4ddaf39e73e80c9c908
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_b5c8be451b55b526063ce859b462f210(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_631651d8b0f96a2da9722c80e221be9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5c8be451b55b526063ce859b462f210
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_b594df6999878a5f3633c4a96a556bc4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dfd5f4abeee673f73fbf182aa784d4cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b594df6999878a5f3633c4a96a556bc4
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dfd5f4abeee673f73fbf182aa784d4cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b594df6999878a5f3633c4a96a556bc4
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_87bacf1654c81504c15bd3ae930a62ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.sum(input_0, input_1, None, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 96, 152, 272], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_470fb447f8c4a46f6b22062043060f14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87bacf1654c81504c15bd3ae930a62ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_6af10e627f8fabec88a4d62fc4fd3c83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03216975927352905, 0.04337479919195175, 0.01657208800315857, 0.09818089008331299], [0.03216975927352905, 0.04337479919195175, 0.01657208800315857, 0.09818089008331299], [0.15343743562698364, 0.3510548174381256, 0.23534007370471954, 0.0949157178401947], [0.12012717127799988, 0.08196210116147995, 0.07514175772666931, 0.4337972104549408], [0.061709702014923096, 0.21900923550128937, 0.12456387281417847, 0.1701676845550537], [0.083442822098732, 0.27063196897506714, 0.05398473143577576, 0.12532399594783783], [0.09118098020553589, 0.2385154515504837, 0.08096741139888763, 0.00832115113735199]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_71197a2f67f34b75265409bbca3b2c9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e7feed7ad207cc9304484b520b5d1c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71197a2f67f34b75265409bbca3b2c9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_05813cd40cbfc822a347738cd3e63a47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5cff72cac6bc423c8e3b7e0d1cfed5f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05813cd40cbfc822a347738cd3e63a47
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5cff72cac6bc423c8e3b7e0d1cfed5f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_05813cd40cbfc822a347738cd3e63a47
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_ffc1ac16bf0f3a4013adab58bd412110(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4849], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d660c248bee498e403e79a9d5e172175(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ffc1ac16bf0f3a4013adab58bd412110
        def get_inputs(self):
            return [
                paddle.uniform([4849], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_bdf7cbcf990267eb7c96b2b8a9db8104(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1206], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c68995ea4d072b4c81d8cfac650bbfbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bdf7cbcf990267eb7c96b2b8a9db8104
        def get_inputs(self):
            return [
                paddle.uniform([1206], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_5fae2df3e98feddbe8e570a97067b942(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2434, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8939cb35c1c65ffc23f1cc0968ba4a5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fae2df3e98feddbe8e570a97067b942
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_764698473c14f18b7d6a912c6e57a70b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d96ab46978e5d2e58dc8d87582ab65b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_764698473c14f18b7d6a912c6e57a70b
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7207fabf5485f94621f6bd146ce4808a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b766297b28d2945758b26ccb1311dec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7207fabf5485f94621f6bd146ce4808a
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_6b766297b28d2945758b26ccb1311dec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7207fabf5485f94621f6bd146ce4808a
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a0172ce0c44eb89cf6c0b77520bd6c99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a73f36534c5f3febd704a5d55d75107
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17413032054901123, 0.02193164825439453, 0.06767009198665619, 0.10391233116388321], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [0.08004461228847504, 0.23684172332286835, 0.0887954980134964, 0.17910811305046082], [0.05183491110801697, 0.12347441911697388, 0.13930287957191467, 0.2521556615829468], [0.21063363552093506, 0.13145971298217773, 0.15847629308700562, 0.02333889901638031]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_434501a68dccfcbe97a3e2547f3bf465(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 2, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a38b7f4d630ea2dc4ee3cc36e726da84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_434501a68dccfcbe97a3e2547f3bf465
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_59f7409ec9197fbd0e416fbeaa8e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c1a76ac269f24645e1336c17fccd09e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 2, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4158663c673624be4e4626e5c737a28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c1a76ac269f24645e1336c17fccd09e6
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4b063ff6a69af8aa6000cabefc30132d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b83a787529cc4b356c6ee0694e664d31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4b063ff6a69af8aa6000cabefc30132d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_9a09b08b281e8007ac75fd0c0ca94a3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09c8ca07f723feaca29d8931349d3aff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a09b08b281e8007ac75fd0c0ca94a3a
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_09c8ca07f723feaca29d8931349d3aff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a09b08b281e8007ac75fd0c0ca94a3a
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_cdcb0b1979c53318fa03c18ae201559d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48e3638323596146d661c1afcc398030(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdcb0b1979c53318fa03c18ae201559d
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_7a301daf54c419d0c8656683e708d5e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c25a64f6ab272878c6589af5b38c98d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a301daf54c419d0c8656683e708d5e5
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_c25a64f6ab272878c6589af5b38c98d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a301daf54c419d0c8656683e708d5e5
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_5570b29155890a52498d0cf11cd9ba75(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08464a6ab95d6ed0e74a6bdea6832d4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5570b29155890a52498d0cf11cd9ba75
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_1881092e3b2be2b02fac877f53da0c37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cef5a8c6bf2150ee52f9aa4967a637e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1881092e3b2be2b02fac877f53da0c37
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_cef5a8c6bf2150ee52f9aa4967a637e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1881092e3b2be2b02fac877f53da0c37
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_1896e33fb6f78c7fc1a95c00ebcd8511(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[247], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ae9dbea1777e2d4eff38c0e422089f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1896e33fb6f78c7fc1a95c00ebcd8511
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4ce4730cb65ba5f15186486e234a1e13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16344afd53d26fc87d987706cbf46fa7
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bf88ce40921ac42d3e835d568e89b99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_53ddadc6c0191e0021b057a229cf28b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2a72ca8ab6367501ee4c932512d33193
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_0941b9d7f3be9adb3a2864993bcc930f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5a149eb8b30ae49efacbc2cda0c0e135(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0941b9d7f3be9adb3a2864993bcc930f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.17320916056632996, 0.08482294529676437, 0.09949600696563721, 0.19735485315322876, 0.2445642054080963, 0.24156497418880463, 0.20749948918819427, 0.0047854408621788025, 0.08320922404527664, 0.196404829621315, 0.07307159155607224, 0.05009898915886879, 0.09655003994703293, 0.08765567094087601, 0.11944684386253357, 0.21975986659526825, 0.08828285336494446, 0.10039164870977402, 0.2237415462732315, 0.07407843321561813], dtype='float32').reshape([20]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_1c4bb66a0f534f165a5e05d86689bc07(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17369], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_18897f0434d03a6d50714c58cc8c2017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c4bb66a0f534f165a5e05d86689bc07
        def get_inputs(self):
            return [
                paddle.uniform([17369], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4ce4730cb65ba5f15186486e234a1e13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16344afd53d26fc87d987706cbf46fa7
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_8ba3ee3da71cc0a1aed44a2b8e1918b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[70], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccb7b521eb62cffd75dd82bf438c0e23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ba3ee3da71cc0a1aed44a2b8e1918b5
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_207615440fc3984110e41d8cf2f093c7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6aeec66aa321951a905de48adf22c28f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_207615440fc3984110e41d8cf2f093c7
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_5aa03630179bd3a9bb53d565812b8606(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 20], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd2b90c9f9a4e6b9381940177635432b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5aa03630179bd3a9bb53d565812b8606
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_4794b7106c8ad82c409964edecf33aa5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd8fe13724332ddadf901141be749d9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4794b7106c8ad82c409964edecf33aa5
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_cd8fe13724332ddadf901141be749d9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4794b7106c8ad82c409964edecf33aa5
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_39016c3c718b96e0ac175b2111e69f48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9c932fce0d50cfbeda7ffc86a5e04dc
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_565ca66f3b0a9b753324b149fe5a6d47(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[551], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f603bb0ab28c9ea6ee6000518fdf7b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_565ca66f3b0a9b753324b149fe5a6d47
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_07e61d79c21463ec91a50d8d700bba2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36759066581726074, 0.1356358379125595, 0.07934285700321198, 0.07276766747236252], [0.37631040811538696, 0.03991866111755371, 0.0015145167708396912, 0.07648283243179321], [0.07664056122303009, 0.06367561221122742, 0.027359262108802795, 0.13421794772148132], [0.07664056122303009, 0.06367561221122742, 0.027359262108802795, 0.13421794772148132], [0.2624303996562958, 0.038322463631629944, 0.20256070792675018, 0.42651233077049255]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_59f7409ec9197fbd0e416fbeaa8e12de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0410b117b3f197d631a80c47ebdac3d
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_0074de74d35da0b8cab911a4dcf4a2ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3800], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_31ed1b4fa037ae275e80ae375bd36016(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0074de74d35da0b8cab911a4dcf4a2ab
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_ce9b79ab6a69d6a3069fb8709fe8d6e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2204], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9937f0ea0b09a1390ed996cf7ba7c491(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce9b79ab6a69d6a3069fb8709fe8d6e1
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_1f9bbf2183b3ee015de34b3331aeafa9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb8045df7c96bb610ef9c4b6f837121b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1f9bbf2183b3ee015de34b3331aeafa9
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_bea34f89e68042dc97707730dd69a5cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 80], dtype='float32'),
                paddle.static.InputSpec(shape=[1], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1eb59b2df749e44b0da330afde6885ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bea34f89e68042dc97707730dd69a5cf
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_12b60c86b6922d550733b02363cc9f3f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b51bfbc8ee6afa10143c4e68d1b60d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12b60c86b6922d550733b02363cc9f3f
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7b51bfbc8ee6afa10143c4e68d1b60d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12b60c86b6922d550733b02363cc9f3f
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dec3c51d75c39a403dbb7f108612149b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55d968ef2e56f0cd92cc3933fae4579e
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_05a95a1b740b7aae6492242b5a41e460(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15342693030834198, 0.16269968450069427, 0.16442686319351196, 0.3595154881477356], [0.0890268087387085, 0.00014069676399230957, 0.25438398122787476, 0.21876737475395203], [0.28361520171165466, 0.2432878464460373, 0.06699240207672119, 0.010794661939144135], [0.15342693030834198, 0.16269968450069427, 0.16442686319351196, 0.3595154881477356], [0.3432466685771942, 0.06492901593446732, 0.25622114539146423, 0.2037186324596405], [0.3985275328159332, 0.0876234769821167, 0.07240909337997437, 0.14636847376823425], [0.3432466685771942, 0.06492901593446732, 0.25622114539146423, 0.2037186324596405]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_c7e13ec6532cbaeb3a524e89f7432391(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a1736a0eb075bf9c1b887ef90311e4e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7e13ec6532cbaeb3a524e89f7432391
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bf88ce40921ac42d3e835d568e89b99b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_480a50fc14d654d19f7648a7975ac7c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [3]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68d8464483249990eba24b88000c1544(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_245d266e9c83766851094bc09d31681d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([4354], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_8f44c800411e46248c5ab294ad2a7609(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44d7f512597727eb5520c947c8ef938f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f44c800411e46248c5ab294ad2a7609
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_256292bbbeccbfef50cde6a645d4e339(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b38365f842993341826d5b8d4c3cc20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_0d7385a53f751734fdb25c5922a528b4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06b5f2c40c7ffda53dbf719dbd5b2d4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_06b5f2c40c7ffda53dbf719dbd5b2d4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824, 2], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3ae2deb2bb7d7a1b3fc706a2677ea49c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.0029696375131607056, 0.008893460966646671]], [[9.824388689594343e-05, 3.097688386333175e-05]], [[0.041273776441812515, 0.03675917908549309]], [[0.0012482624733820558, 0.044441789388656616]], [[0.03526579588651657, 0.1963946372270584]], [[0.08096426725387573, 0.09441264718770981]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_4acb156cd45feee0dfd8eea0cb4cf4a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.09834550321102142, 0.06611254066228867]], [[0.03077700175344944, 0.05494692549109459]], [[0.02604793757200241, 0.03245082497596741]], [[0.017015304416418076, 0.004119689576327801]], [[0.0008794770110398531, 0.013181345537304878]], [[0.014345181174576283, 0.04793524742126465]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_0658bccf0655ddfdcf8f79a71c37ad32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f6dfc15b1fb7bbb68c919cfb6c2b9aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2279099076986313, 0.007394756190478802, 0.1952025145292282, 0.08338708430528641, 0.12485320121049881, 0.07469327002763748, 0.1628420203924179, 0.09415122121572495, 0.195261150598526, 0.008562020026147366, 0.22482548654079437, 0.20597805082798004, 0.009456115774810314, 0.18998557329177856, 0.06557527929544449, 0.017191434279084206], dtype='float32').reshape([16]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d50ec4cda7705fc6aa4100fb3ffe950f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7a8336d7f1c416ef378969f4b50c1e22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([150], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f33ce692a8dfe343d0dd1ab1a4948a09(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_56cd1951efaed104b1d802639a1ff174(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [-1]
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0cc5a7577b13915081ca2c6cc9003cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_a5df93eaa39b4ecf70d32b9ce65454e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a5df93eaa39b4ecf70d32b9ce65454e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1774, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_15b4fc96c4bbcca1ff279c3bcec675ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_47b77dbae873a7a599866da381e89958(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.059835709631443024, 0.2431793510913849, 0.06668603420257568, 0.09330688416957855], [0.030676662921905518, 0.4153245687484741, 0.09938794374465942, 0.10088853538036346], [0.041919320821762085, 0.11580070853233337, 0.13541805744171143, 0.10899001359939575], [0.19998770952224731, 0.3514717221260071, 0.011652082204818726, 0.3247314393520355], [0.10408538579940796, 0.18593794107437134, 0.10364016890525818, 0.07083243876695633]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_0d14d0bb64afe96837ef208b6f8c47d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 4, 16, 49, 56, 56], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_182f948aa63389f2bad5cb602a04dd24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07945215702056885, 0.19198602437973022, 0.09651744365692139, 0.051396384835243225], [0.1376815140247345, 0.03616875410079956, 0.030297398567199707, 0.08579754829406738], [0.06492412090301514, 0.02363431453704834, 0.03141921013593674, 0.15331938862800598], [0.1376815140247345, 0.03616875410079956, 0.030297398567199707, 0.08579754829406738], [0.06492412090301514, 0.02363431453704834, 0.03141921013593674, 0.15331938862800598]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9db847cbef871c7761bc5b6008fa66be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f44c800411e46248c5ab294ad2a7609
        def get_inputs(self):
            return [
                paddle.uniform([1, 21824, 15], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dd8cdd9c49218062af81528405dbcab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3b1a139dc69ced1b95704658ec674914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3b1a139dc69ced1b95704658ec674914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([5454, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5d1d79c28a9c7928d286c275ed6e74ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10889467597007751, 0.25807124376296997, 0.14413365721702576, 0.4194449782371521], [0.3009010851383209, 0.18025171756744385, 0.11982408165931702, 0.061148613691329956], [0.36769217252731323, 0.1304820477962494, 0.4372129440307617, 0.006102010607719421], [0.3009010851383209, 0.18025171756744385, 0.11982408165931702, 0.061148613691329956], [0.36769217252731323, 0.1304820477962494, 0.4372129440307617, 0.006102010607719421], [0.1060771495103836, 0.28093963861465454, 0.11857184767723083, 0.316922664642334], [0.1060771495103836, 0.28093963861465454, 0.11857184767723083, 0.316922664642334]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3457b8993b7ff51e693c67e3810f90c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f93e291a0a0563f6318d433d008a3e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bd224957295b1046aadaab7d36465e68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_0cc5a7577b13915081ca2c6cc9003cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3839dd1c05f5ff2e9e949b613d4b268f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3839dd1c05f5ff2e9e949b613d4b268f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1722, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_15b4fc96c4bbcca1ff279c3bcec675ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([15200], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_470b14361cdfd02ed61680cb2b16ed7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_68a8313961ddf07a409f5503bf1765e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.1698136031627655, 0.1802951991558075, 0.08069563657045364, 0.16915643215179443, 0.21542100608348846, 0.16946832835674286, 0.023178303614258766, 0.1980338990688324, 0.018291577696800232, 0.10876372456550598, 0.07114827632904053, 0.08529523015022278, 0.12002154439687729, 0.2009904384613037, 0.17093011736869812, 0.2671310007572174, 0.18313130736351013, 0.20361387729644775, 0.12102005630731583, 0.09615907818078995, 0.24224619567394257, 0.2328210026025772, 0.2187272012233734, 0.12140653282403946], dtype='float32').reshape([24]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_37065dc4a8f2d31a5e1a9719c268dad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_caa43e8cdf790ddc41a6381e8e55cc08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_caa43e8cdf790ddc41a6381e8e55cc08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1518, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a04ba462c5ea746d0ad25c55362be7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_54287def0a0361bfdf5d42132bfb5615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.03049149177968502, 0.016200125217437744, 0.21287184953689575, 0.11618273705244064], dtype='float32').reshape([4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_6c98204e89087ab13f57bdb470a32f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22957313060760498, 0.17837467789649963, 0.016854897141456604, 0.40951916575431824], [0.030538499355316162, 0.026057451963424683, 0.00427737832069397, 0.09727845340967178], [0.25486868619918823, 0.3133251667022705, 0.23783250153064728, 0.25276681780815125], [0.15342719852924347, 0.05401526391506195, 0.19605214893817902, 0.08178121596574783], [0.15342719852924347, 0.05401526391506195, 0.19605214893817902, 0.08178121596574783], [0.25486868619918823, 0.3133251667022705, 0.23783250153064728, 0.25276681780815125]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4025f75102b536bced024bfa19bd23e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21438781917095184, 0.07429340481758118, 0.2580089867115021, 0.2164541482925415], [0.13806931674480438, 0.42606988549232483, 0.00037895888090133667, 0.46145597100257874], [0.12322920560836792, 0.12052872776985168, 0.28811296820640564, 0.20321156084537506], [0.05649891495704651, 0.06143069267272949, 0.13873469829559326, 0.31833985447883606], [0.21438781917095184, 0.07429340481758118, 0.2580089867115021, 0.2164541482925415]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f8e2607bcbf9d2c58358ef94eb11743c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a04ba462c5ea746d0ad25c55362be7ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_cf9ac93c9f090007fabe75db14b28a4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04451256990432739, 0.19473513960838318, 0.36136090755462646, 0.024676114320755005], [0.04359281063079834, 0.4449837803840637, 0.11287054419517517, 0.006923645734786987], [0.2663431763648987, 0.1137508675456047, 0.3312119245529175, 0.14764076471328735], [0.16015861928462982, 0.13828155398368835, 0.009508371353149414, 0.2052965611219406]], dtype='float32').reshape([4, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_754fed2f70aa3a5eafce174bc4d1acc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [2, 3]
            return paddle._C_ops.sum(input_0, input_1, None, True)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c126851cb4a52059651574ef86ac8238(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_754fed2f70aa3a5eafce174bc4d1acc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 34], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_470b14361cdfd02ed61680cb2b16ed7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3f6b6ec45b6f6f1adec1f2d47ed27344(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_fab3b559fedb220a6c8a2b996b8a03a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([950], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_4fe8c4cbd19791505ae34c8c575dde43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([8816], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_c8f45b4218d8271d9ded954b7dd0a168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5b1230bddeec2cb34e605449dcb0b015(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5b1230bddeec2cb34e605449dcb0b015(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2133, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e5fe07c8ea3b870bf4ced31786b6f6e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_754fed2f70aa3a5eafce174bc4d1acc9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 152, 272], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([2, 3], dtype='int64').reshape([2]),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b271d5277059d96f2b5ff930b0e4dbae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03216975927352905, 0.04337479919195175, 0.01657208800315857, 0.09818089008331299], [0.03216975927352905, 0.04337479919195175, 0.01657208800315857, 0.09818089008331299], [0.15343743562698364, 0.3510548174381256, 0.23534007370471954, 0.0949157178401947], [0.12012717127799988, 0.08196210116147995, 0.07514175772666931, 0.4337972104549408], [0.061709702014923096, 0.21900923550128937, 0.12456387281417847, 0.1701676845550537], [0.083442822098732, 0.27063196897506714, 0.05398473143577576, 0.12532399594783783], [0.09118098020553589, 0.2385154515504837, 0.08096741139888763, 0.00832115113735199]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_c704ff20cf334773d69e1fe9e0551537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_dd0627d598e329d841ebc7f0268c4101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dd0627d598e329d841ebc7f0268c4101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4631, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_77d707cce6a2f5fe5cd8f18cd213c5a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([4849], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_c5a040c030fbe3fe5235f3da326949b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([1206], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_17e378482dfb8dfea6625685e7d9f5f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f44c800411e46248c5ab294ad2a7609
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_3e39b0e78205354e9ad07446c4f4f263(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_b6550d29ad3f764e498a3bbcc9403621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b6550d29ad3f764e498a3bbcc9403621(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1039, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_64dd8c1f8c6e5051e32e194b7c62a976(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17413032054901123, 0.02193164825439453, 0.06767009198665619, 0.10391233116388321], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [0.08004461228847504, 0.23684172332286835, 0.0887954980134964, 0.17910811305046082], [0.05183491110801697, 0.12347441911697388, 0.13930287957191467, 0.2521556615829468], [0.21063363552093506, 0.13145971298217773, 0.15847629308700562, 0.02333889901638031]], dtype='float32').reshape([6, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_dc7010f1952ae29f58ceb9df3aaa9d2a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_121c97acd5bb5015337e1d111daf0125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_60ee61f1bce7bde445ad595c8a1ea2c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_195dfff932a78f77b0e0d5458aafdf63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_404e8c5c8734467ff830e417129eb863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_404e8c5c8734467ff830e417129eb863(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2318, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_097597739e1363c15d52fbf033aa2fde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5a7c51a8c3a2b4343e33ffb3a2137400(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5a7c51a8c3a2b4343e33ffb3a2137400(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2961, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8af3d5279b78e0b2367c7631e8e0cc67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_22e0786b5673e7eba5fbff5d3f068353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_22e0786b5673e7eba5fbff5d3f068353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([3739, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_9298619f4ded9d6f087045ba05dd25be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([247], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7484cfd2b20d4fe43c0bd3a213dcc89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e8da34dc8b39359ce8dcd4f186d1578a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_f93e291a0a0563f6318d433d008a3e03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_59c57cb1a829bda23ccc94cc119dbb1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.17320916056632996, 0.08482294529676437, 0.09949600696563721, 0.19735485315322876, 0.2445642054080963, 0.24156497418880463, 0.20749948918819427, 0.0047854408621788025, 0.08320922404527664, 0.196404829621315, 0.07307159155607224, 0.05009898915886879, 0.09655003994703293, 0.08765567094087601, 0.11944684386253357, 0.21975986659526825, 0.08828285336494446, 0.10039164870977402, 0.2237415462732315, 0.07407843321561813], dtype='float32').reshape([20]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_99050c7d2672f539eb2b169bdd439bf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([17369], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7484cfd2b20d4fe43c0bd3a213dcc89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([10, 8, 16, 49, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_6af6394c2f7d3c64dea122d9c6ca9cf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([70], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8f9179bb084f279eace036a308aae8e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b175020cb17eda1819a96b0975ffe213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 20], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_82827445cb2338f7be53aac049fc87d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_82827445cb2338f7be53aac049fc87d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2013, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_098b990a49e58d2f4a11cacc5eeabb30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2, 16, 9, 112, 112], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e47361bbc6bdcaef6b8b58be86c71fee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([551], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_101dee268b37d0e2fc4bb19d6e764686(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36759066581726074, 0.1356358379125595, 0.07934285700321198, 0.07276766747236252], [0.37631040811538696, 0.03991866111755371, 0.0015145167708396912, 0.07648283243179321], [0.07664056122303009, 0.06367561221122742, 0.027359262108802795, 0.13421794772148132], [0.07664056122303009, 0.06367561221122742, 0.027359262108802795, 0.13421794772148132], [0.2624303996562958, 0.038322463631629944, 0.20256070792675018, 0.42651233077049255]], dtype='float32').reshape([5, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_121c97acd5bb5015337e1d111daf0125(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 16, 49, 7, 7], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ab8be978244be906c6ccd93b307aa8dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([3800], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_02254a92124454906188587ca3ae5831(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([2204], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_26a0b39748adc5fcf982520a9ec38aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_be7add0de564f1e428d13208ba3eb4b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_56cd1951efaed104b1d802639a1ff174
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 80], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_37baf4f2f7d25514e350502a077947b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_37baf4f2f7d25514e350502a077947b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4177, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bfba72c4beabf18de38e574d3ef29857(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7e266419d7917f55c51f62875dca50cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.15342693030834198, 0.16269968450069427, 0.16442686319351196, 0.3595154881477356], [0.0890268087387085, 0.00014069676399230957, 0.25438398122787476, 0.21876737475395203], [0.28361520171165466, 0.2432878464460373, 0.06699240207672119, 0.010794661939144135], [0.15342693030834198, 0.16269968450069427, 0.16442686319351196, 0.3595154881477356], [0.3432466685771942, 0.06492901593446732, 0.25622114539146423, 0.2037186324596405], [0.3985275328159332, 0.0876234769821167, 0.07240909337997437, 0.14636847376823425], [0.3432466685771942, 0.06492901593446732, 0.25622114539146423, 0.2037186324596405]], dtype='float32').reshape([7, 4]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_907f63a2d245f9f81f590c3b8416252a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e8da34dc8b39359ce8dcd4f186d1578a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_420c675b9b1bef15a3e9b34b24fc8c7e
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 16, 49, 14, 14], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    

if __name__ == '__main__':
    unittest.main()