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


    class TestPrimitiveOp_b3d7e32acfce35b4688986060f92fa5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([4329], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_bd1e7fcb6bf45e0f18bc9e4544a15e9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.01666249893605709, 0.003314714413136244]], [[0.004053822718560696, 0.0006664209067821503]], [[0.0019431465771049261, 0.003587140701711178]], [[0.029846083372831345, 0.02684077061712742]], [[0.005597985349595547, 0.0021670570131391287]], [[0.14419999718666077, 0.05131029710173607]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_61e6c6ecfc318b3018b9f4f92855351a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8cad2416033e73c293f2e492f0cc60c6
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.0019043288193643093, 0.00015663662634324282]], [[0.013697666116058826, 0.001827340922318399]], [[0.0012445304309949279, 0.09536285698413849]], [[0.01113487221300602, 0.046624600887298584]], [[0.06389683485031128, 0.07109943777322769]], [[0.004716935567557812, 0.014792771078646183]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_5423ac204aa4411d5635530835f91541(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.20323149859905243, 0.09429150074720383, 0.07140893489122391, 0.1901603490114212, 0.16385573148727417, 0.09601243585348129, 0.1287786364555359, 0.23864251375198364, 0.07157084345817566, 0.2625637948513031, 0.20802748203277588, 0.1473717987537384, 0.26525965332984924, 0.2729385793209076, 0.06359413266181946, 0.20311571657657623], dtype='float32').reshape([16]),
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


    class TestPrimitiveOp_0755b8d42da93a8e9ab68787adc292ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_0755b8d42da93a8e9ab68787adc292ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0dc71acb0513080141afac1a9190ce61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.016139358282089233, 0.024570047855377197, 0.1798972338438034, 0.38466477394104004], [0.3155452609062195, 0.11460816860198975, 0.05521531403064728, 0.06962801516056061], [0.045381829142570496, 0.20968413352966309, 0.11353392899036407, 0.1670221984386444], [0.3187718987464905, 0.2138129323720932, 0.054648011922836304, 0.12318618595600128], [0.14328435063362122, 0.12580473721027374, 0.005007922649383545, 0.018766164779663086]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_c2d0a35711a1ebc42a793794cc105559(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17177636921405792, 0.1657608598470688, 0.2102193832397461, 0.09437558054924011], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, 0.017763465642929077, 0.1887105405330658], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, 0.017763465642929077, 0.1887105405330658]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_2b5cb2e117e8737fc2501681ebf1cbf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_2b5cb2e117e8737fc2501681ebf1cbf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e851cb54d693bad1579706b32dbf31b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0048261284828186035, 0.07155326008796692, 0.013605579733848572, 0.051180481910705566], [0.14474661648273468, 0.2696493864059448, 0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, 0.3962211310863495, 0.15296784043312073], [0.14474661648273468, 0.2696493864059448, 0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, 0.3962211310863495, 0.15296784043312073], [0.22924979031085968, 0.1217564269900322, 0.07167693972587585, 0.07170835137367249], [0.22924979031085968, 0.1217564269900322, 0.07167693972587585, 0.07170835137367249]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_d326de4f925e83cb155fc4c7053f1748(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d326de4f925e83cb155fc4c7053f1748(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_eb11815d67b225bdf0e1833b6fb9ceb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.13257187604904175, 0.18898636102676392, 0.11631551384925842, 0.19848103821277618, 0.05323219299316406, 0.1827613115310669, 0.2413797378540039, 0.21782590448856354, 0.04551813006401062, 0.05982864275574684, 0.13052749633789062, 0.024526027962565422, 0.22515442967414856, 0.21618947386741638, 0.008748939260840416, 0.002882672706618905, 0.04029916226863861, 0.21624203026294708, 0.21809883415699005, 0.1499156802892685, 0.0032108670566231012, 0.17229510843753815, 0.06436792761087418, 0.251303493976593], dtype='float32').reshape([24]),
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


    class TestPrimitiveOp_682b948aef60c538849c695fbd221022(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_682b948aef60c538849c695fbd221022(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2990128469e15a02ff18735f0e5a0e5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.08394922316074371, 0.24046628177165985, 0.1896965354681015, 0.04368181526660919], dtype='float32').reshape([4]),
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


    class TestPrimitiveOp_db525df132a8d3906ff452e48edd0ee4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07304361462593079, 0.10842274129390717, 0.23478034138679504, 0.023416876792907715], [0.03650416433811188, 0.2645467519760132, 0.3254188299179077, 0.08861064910888672], [0.26510629057884216, 0.05608522891998291, 0.21578019857406616, 0.10049128532409668], [0.07779538631439209, 0.06618337333202362, 0.022951990365982056, 0.24318337440490723], [0.07779538631439209, 0.06618337333202362, 0.022951990365982056, 0.24318337440490723], [0.26510629057884216, 0.05608522891998291, 0.21578019857406616, 0.10049128532409668]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_29ef43010a0bc595637fcb92a5b107c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.013753771781921387, 0.2868351638317108, 0.03862294554710388, 0.19162918627262115], [0.37577491998672485, 0.05430358648300171, 0.24721841514110565, 0.08289992064237595], [0.13008549809455872, 0.14730504155158997, 0.15147680044174194, 0.33067959547042847], [0.2519821524620056, 0.363572895526886, 0.131232351064682, 0.2127522975206375], [0.013753771781921387, 0.2868351638317108, 0.03862294554710388, 0.19162918627262115]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_6982a6287f297890619f66642764148f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10226257890462875, 0.13534961640834808, 0.08023402094841003, 0.039990901947021484], [0.29901987314224243, 0.09267288446426392, 0.20913521945476532, 0.4559893012046814], [0.2016167938709259, 0.13266520202159882, 0.10226882994174957, 0.20165568590164185], [0.008473038673400879, 0.02449806034564972, 0.06644713133573532, 0.26101258397102356]], dtype='float32').reshape([4, 4]),
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


    class TestPrimitiveOp_30fe579a9262394daabc791045bc1c75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_30fe579a9262394daabc791045bc1c75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_51ff2eb05dfdfb15740b32c1b5064b33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1575099527835846, 0.26694709062576294, 0.34720781445503235, 0.21742114424705505], [0.1575099527835846, 0.26694709062576294, 0.34720781445503235, 0.21742114424705505], [0.1509353071451187, 0.1447482705116272, 0.2221795916557312, 0.09455010294914246], [0.12404389679431915, 0.06935502588748932, 0.12357345223426819, 0.3634084165096283], [0.013999328017234802, 0.39129024744033813, 0.18125993013381958, 0.20690134167671204], [0.006215885281562805, 0.10090631246566772, 0.3123472034931183, 0.14281310141086578], [0.3736182451248169, 0.07743585109710693, 0.2678108513355255, 0.25866973400115967]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_17f087d75a41f5e0dc3ab06510b31339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_17f087d75a41f5e0dc3ab06510b31339(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_80db884326de4b91f4a76160ce662acb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([4851], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_74eb40deb995817da8dbad0dcbbc27e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([1224], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b4cd07c0d963cdf254639a8fefcb937b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b4cd07c0d963cdf254639a8fefcb937b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_90f2d105e3b0334c0fb32e7cbbb02fdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1208452433347702, 0.1525149941444397, 0.18495747447013855, 0.21681168675422668], [0.016017157584428787, 0.0064589232206344604, 0.21185481548309326, 0.15630508959293365], [0.016017157584428787, 0.0064589232206344604, 0.21185481548309326, 0.15630508959293365], [0.35850846767425537, 0.1164940744638443, 0.17222779989242554, 0.30740097165107727], [0.02179768681526184, 0.04435417056083679, 0.2214246392250061, 0.19965055584907532], [0.01850026845932007, 0.061271607875823975, 0.1330414116382599, 0.09196221828460693]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_7263e23a2b9aafc7be2aa9d0457adac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7263e23a2b9aafc7be2aa9d0457adac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8b74615dd186b17597641a0080bc741a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8b74615dd186b17597641a0080bc741a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_6fe8d44967acb1b3262dde7581554a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_6fe8d44967acb1b3262dde7581554a02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_649ba6fd9150673b14e7861bc89f25ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.04458913579583168, 0.21033427119255066, 0.07286225259304047, 0.03924795612692833, 0.12533174455165863, 0.22624242305755615, 0.09203226864337921, 0.15758462250232697, 0.02457907423377037, 0.21276026964187622, 0.06444792449474335, 0.07556305080652237, 0.19715236127376556, 0.07277515530586243, 0.18578313291072845, 0.08678262680768967, 0.09069261699914932, 0.12000761181116104, 0.2193279266357422, 0.14802876114845276], dtype='float32').reshape([20]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_0e08e61a74835772540da55085c34e43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7d8e8ba5ab3e88ab0ae4e19b2a13dccf
        def get_inputs(self):
            return [
                paddle.uniform([17475], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e30a1bae1c9d7b2141f34cffaeb1562b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e30a1bae1c9d7b2141f34cffaeb1562b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_3257f50a3c5a13599d1713e6411e3cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0021021366119384766, 0.10599559545516968, 0.1741461157798767, 0.11614590883255005], [0.2877083122730255, 0.18331629037857056, 0.15134862065315247, 0.30544114112854004], [0.3105199337005615, 0.16365931928157806, 0.19212684035301208, 0.06973066926002502], [0.3105199337005615, 0.16365931928157806, 0.19212684035301208, 0.06973066926002502], [0.2566255033016205, 0.30786266922950745, 0.02622893452644348, 0.3976331949234009]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_5aabb471a0917e8bb940875367711789(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_5aabb471a0917e8bb940875367711789(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_895fd06472fffb6cb451bb012c824929
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_512f5fd4923c02759b27f23b9cd3f4e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ee8c8517744412d8bc6217d5aa7ce6b
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.026198573410511017, 0.122957244515419, 0.025363638997077942, 0.051180094480514526], [0.26046913862228394, 0.16602113842964172, 0.2520030736923218, 0.0859014093875885], [0.27233195304870605, 0.3711124062538147, 0.03736155107617378, 0.2736709713935852], [0.026198573410511017, 0.122957244515419, 0.025363638997077942, 0.051180094480514526], [0.0075002312660217285, 0.25931796431541443, 0.013504356145858765, 0.38065314292907715], [0.3905479311943054, 0.20227286219596863, 0.22368675470352173, 0.24777323007583618], [0.0075002312660217285, 0.25931796431541443, 0.013504356145858765, 0.38065314292907715]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_49178b619ca5799ccb6bdf56b47cd9f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4329], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5276dacd3caa20b29ce0ee2e3aaaa194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49178b619ca5799ccb6bdf56b47cd9f9
        def get_inputs(self):
            return [
                paddle.uniform([4329], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_0585231933fee57be29aca2378ee89c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01bba0a914ebd1f491ac9c5013dccca5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.01666249893605709, 0.003314714413136244]], [[0.004053822718560696, 0.0006664209067821503]], [[0.0019431465771049261, 0.003587140701711178]], [[0.029846083372831345, 0.02684077061712742]], [[0.005597985349595547, 0.0021670570131391287]], [[0.14419999718666077, 0.05131029710173607]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_e8cfa88a9988c334247b7da70fa4fe5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01bba0a914ebd1f491ac9c5013dccca5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.0019043288193643093, 0.00015663662634324282]], [[0.013697666116058826, 0.001827340922318399]], [[0.0012445304309949279, 0.09536285698413849]], [[0.01113487221300602, 0.046624600887298584]], [[0.06389683485031128, 0.07109943777322769]], [[0.004716935567557812, 0.014792771078646183]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_be3f8e7afa15d3a8913ed6543eddc281(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fc6813c50b3e0aa20e38d15ae3e4b43
        def get_inputs(self):
            return [
                paddle.to_tensor([0.20323149859905243, 0.09429150074720383, 0.07140893489122391, 0.1901603490114212, 0.16385573148727417, 0.09601243585348129, 0.1287786364555359, 0.23864251375198364, 0.07157084345817566, 0.2625637948513031, 0.20802748203277588, 0.1473717987537384, 0.26525965332984924, 0.2729385793209076, 0.06359413266181946, 0.20311571657657623], dtype='float32').reshape([16]),
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


    
    class PrimitiveOp_b394c962afee84019b73a73acd51115f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e3e4c0e8b21f2a718e616b2a3944583d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b394c962afee84019b73a73acd51115f
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_e3e4c0e8b21f2a718e616b2a3944583d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b394c962afee84019b73a73acd51115f
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_bd50aad49c760b4e1b1dfad59bd32055(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.016139358282089233, 0.024570047855377197, 0.1798972338438034, 0.38466477394104004], [0.3155452609062195, 0.11460816860198975, 0.05521531403064728, 0.06962801516056061], [0.045381829142570496, 0.20968413352966309, 0.11353392899036407, 0.1670221984386444], [0.3187718987464905, 0.2138129323720932, 0.054648011922836304, 0.12318618595600128], [0.14328435063362122, 0.12580473721027374, 0.005007922649383545, 0.018766164779663086]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_3f26dcc8759478f8247f60b95cf6ee06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17177636921405792, 0.1657608598470688, 0.2102193832397461, 0.09437558054924011], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, 0.017763465642929077, 0.1887105405330658], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, 0.017763465642929077, 0.1887105405330658]], dtype='float32').reshape([5, 4]),
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


    
    class PrimitiveOp_1fac0aef4b39fb5b9a68d390fa8809ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af2c7569440875519b439bbb065042a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fac0aef4b39fb5b9a68d390fa8809ce
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_af2c7569440875519b439bbb065042a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1fac0aef4b39fb5b9a68d390fa8809ce
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8d56a9f162faa5ca68849604e715c8f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0048261284828186035, 0.07155326008796692, 0.013605579733848572, 0.051180481910705566], [0.14474661648273468, 0.2696493864059448, 0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, 0.3962211310863495, 0.15296784043312073], [0.14474661648273468, 0.2696493864059448, 0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, 0.3962211310863495, 0.15296784043312073], [0.22924979031085968, 0.1217564269900322, 0.07167693972587585, 0.07170835137367249], [0.22924979031085968, 0.1217564269900322, 0.07167693972587585, 0.07170835137367249]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_4bf345b37846d8363799f7decf9134dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_582844d5d8521b427bcf8b64f2ff78b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bf345b37846d8363799f7decf9134dc
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_582844d5d8521b427bcf8b64f2ff78b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bf345b37846d8363799f7decf9134dc
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d00b0bb0e20aed479fdf648a4cad01e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_abf701896c1b5c2cda506d95ebfa0da8
        def get_inputs(self):
            return [
                paddle.to_tensor([0.13257187604904175, 0.18898636102676392, 0.11631551384925842, 0.19848103821277618, 0.05323219299316406, 0.1827613115310669, 0.2413797378540039, 0.21782590448856354, 0.04551813006401062, 0.05982864275574684, 0.13052749633789062, 0.024526027962565422, 0.22515442967414856, 0.21618947386741638, 0.008748939260840416, 0.002882672706618905, 0.04029916226863861, 0.21624203026294708, 0.21809883415699005, 0.1499156802892685, 0.0032108670566231012, 0.17229510843753815, 0.06436792761087418, 0.251303493976593], dtype='float32').reshape([24]),
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


    
    class PrimitiveOp_34dc4f9ccb4563b4cdcc1a57d75442fe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_26e558baa225cafd1bc9f4feb3a66a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34dc4f9ccb4563b4cdcc1a57d75442fe
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_26e558baa225cafd1bc9f4feb3a66a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34dc4f9ccb4563b4cdcc1a57d75442fe
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_3935ce4e4ef18b31a28188bef8d00cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fc26a21b27eba3cdde678c037676765a
        def get_inputs(self):
            return [
                paddle.to_tensor([0.08394922316074371, 0.24046628177165985, 0.1896965354681015, 0.04368181526660919], dtype='float32').reshape([4]),
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


    class TestPrimitiveOp_55f7e3a0340296f93b64a256234046c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a73f36534c5f3febd704a5d55d75107
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07304361462593079, 0.10842274129390717, 0.23478034138679504, 0.023416876792907715], [0.03650416433811188, 0.2645467519760132, 0.3254188299179077, 0.08861064910888672], [0.26510629057884216, 0.05608522891998291, 0.21578019857406616, 0.10049128532409668], [0.07779538631439209, 0.06618337333202362, 0.022951990365982056, 0.24318337440490723], [0.07779538631439209, 0.06618337333202362, 0.022951990365982056, 0.24318337440490723], [0.26510629057884216, 0.05608522891998291, 0.21578019857406616, 0.10049128532409668]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_effce44594446a60ebb2fb8c7335a9f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.013753771781921387, 0.2868351638317108, 0.03862294554710388, 0.19162918627262115], [0.37577491998672485, 0.05430358648300171, 0.24721841514110565, 0.08289992064237595], [0.13008549809455872, 0.14730504155158997, 0.15147680044174194, 0.33067959547042847], [0.2519821524620056, 0.363572895526886, 0.131232351064682, 0.2127522975206375], [0.013753771781921387, 0.2868351638317108, 0.03862294554710388, 0.19162918627262115]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_6248266e232fbfe488c195ed8b0ca7b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4b14492b5693ed40c5ffef38cd64f08
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10226257890462875, 0.13534961640834808, 0.08023402094841003, 0.039990901947021484], [0.29901987314224243, 0.09267288446426392, 0.20913521945476532, 0.4559893012046814], [0.2016167938709259, 0.13266520202159882, 0.10226882994174957, 0.20165568590164185], [0.008473038673400879, 0.02449806034564972, 0.06644713133573532, 0.26101258397102356]], dtype='float32').reshape([4, 4]),
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


    
    class PrimitiveOp_f35c717179a76bc95161288a3d184b58(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60dcb75ae34eb914db5dab8bbf846a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f35c717179a76bc95161288a3d184b58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_60dcb75ae34eb914db5dab8bbf846a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f35c717179a76bc95161288a3d184b58
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_ae17727aff8f91ad620d827b7a2520b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1575099527835846, 0.26694709062576294, 0.34720781445503235, 0.21742114424705505], [0.1575099527835846, 0.26694709062576294, 0.34720781445503235, 0.21742114424705505], [0.1509353071451187, 0.1447482705116272, 0.2221795916557312, 0.09455010294914246], [0.12404389679431915, 0.06935502588748932, 0.12357345223426819, 0.3634084165096283], [0.013999328017234802, 0.39129024744033813, 0.18125993013381958, 0.20690134167671204], [0.006215885281562805, 0.10090631246566772, 0.3123472034931183, 0.14281310141086578], [0.3736182451248169, 0.07743585109710693, 0.2678108513355255, 0.25866973400115967]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_eb83385f84010eb94296c1ebdc7d4ebe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_613efcf1fcf59ac2dd73038c69a7bbd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83385f84010eb94296c1ebdc7d4ebe
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_613efcf1fcf59ac2dd73038c69a7bbd9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83385f84010eb94296c1ebdc7d4ebe
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_cfdbd6818888cda63cd3ea58044d2214(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4851], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_670634db96b80e932daec158f0f5bf26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cfdbd6818888cda63cd3ea58044d2214
        def get_inputs(self):
            return [
                paddle.uniform([4851], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_10dba7cc2f5408b39377b320a23e3e9d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1224], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83abbfa699c50944eeab4750a1b17ff7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_10dba7cc2f5408b39377b320a23e3e9d
        def get_inputs(self):
            return [
                paddle.uniform([1224], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_d87afcdc0c1ab5eeccf4a36597564705(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be036d5e75701b52ce5a8045bb701e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d87afcdc0c1ab5eeccf4a36597564705
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_be036d5e75701b52ce5a8045bb701e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d87afcdc0c1ab5eeccf4a36597564705
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c87fef698d7ecd1345223496ace21223(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a73f36534c5f3febd704a5d55d75107
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1208452433347702, 0.1525149941444397, 0.18495747447013855, 0.21681168675422668], [0.016017157584428787, 0.0064589232206344604, 0.21185481548309326, 0.15630508959293365], [0.016017157584428787, 0.0064589232206344604, 0.21185481548309326, 0.15630508959293365], [0.35850846767425537, 0.1164940744638443, 0.17222779989242554, 0.30740097165107727], [0.02179768681526184, 0.04435417056083679, 0.2214246392250061, 0.19965055584907532], [0.01850026845932007, 0.061271607875823975, 0.1330414116382599, 0.09196221828460693]], dtype='float32').reshape([6, 4]),
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


    
    class PrimitiveOp_f96df294e7f51ad2e3e8e04b5c3ad1ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8cb168890c03c2f6f460063d5953aafa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f96df294e7f51ad2e3e8e04b5c3ad1ba
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8cb168890c03c2f6f460063d5953aafa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f96df294e7f51ad2e3e8e04b5c3ad1ba
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_849dc7fd0dfc921a1df95b4e1c5f701b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e311c37ae518a80e8498d8d59d8cebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_849dc7fd0dfc921a1df95b4e1c5f701b
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_1e311c37ae518a80e8498d8d59d8cebb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_849dc7fd0dfc921a1df95b4e1c5f701b
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_0268028e63bcfb1e77594e70b6a835a7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd4fac4dbc2933e54750635725633820(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0268028e63bcfb1e77594e70b6a835a7
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_bd4fac4dbc2933e54750635725633820(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0268028e63bcfb1e77594e70b6a835a7
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_edb526a2d73d05e9ceb3b89a1132588f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0941b9d7f3be9adb3a2864993bcc930f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.04458913579583168, 0.21033427119255066, 0.07286225259304047, 0.03924795612692833, 0.12533174455165863, 0.22624242305755615, 0.09203226864337921, 0.15758462250232697, 0.02457907423377037, 0.21276026964187622, 0.06444792449474335, 0.07556305080652237, 0.19715236127376556, 0.07277515530586243, 0.18578313291072845, 0.08678262680768967, 0.09069261699914932, 0.12000761181116104, 0.2193279266357422, 0.14802876114845276], dtype='float32').reshape([20]),
                paddle.to_tensor([], dtype='int64'),
            ]


    
    class PrimitiveOp_5fe8595add5eb91a75513c2b18daf476(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17475], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94c6bda241bd8dbcdb6335e1b277abf0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fe8595add5eb91a75513c2b18daf476
        def get_inputs(self):
            return [
                paddle.uniform([17475], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_5e7b285b0258fa29644b3ab718f620cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81d2485ddc88f26b87e4ac771ee0e3ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e7b285b0258fa29644b3ab718f620cd
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_81d2485ddc88f26b87e4ac771ee0e3ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e7b285b0258fa29644b3ab718f620cd
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4035c29885b6e522c26e379f34de28e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb83881230c2be7127771fc7af7c350a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0021021366119384766, 0.10599559545516968, 0.1741461157798767, 0.11614590883255005], [0.2877083122730255, 0.18331629037857056, 0.15134862065315247, 0.30544114112854004], [0.3105199337005615, 0.16365931928157806, 0.19212684035301208, 0.06973066926002502], [0.3105199337005615, 0.16365931928157806, 0.19212684035301208, 0.06973066926002502], [0.2566255033016205, 0.30786266922950745, 0.02622893452644348, 0.3976331949234009]], dtype='float32').reshape([5, 4]),
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


    
    class PrimitiveOp_a18cb72212024f3c81db35691a7110b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = []
            return paddle._C_ops.sum(input_0, input_1, None, False)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a92516996e0ef01cd682af24d318eb2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a18cb72212024f3c81db35691a7110b3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a92516996e0ef01cd682af24d318eb2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a18cb72212024f3c81db35691a7110b3
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_85e95ca815c0a5525dd3cd64904eb438(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_19df5c8bf10a4c10da3dea6b19879c56
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.026198573410511017, 0.122957244515419, 0.025363638997077942, 0.051180094480514526], [0.26046913862228394, 0.16602113842964172, 0.2520030736923218, 0.0859014093875885], [0.27233195304870605, 0.3711124062538147, 0.03736155107617378, 0.2736709713935852], [0.026198573410511017, 0.122957244515419, 0.025363638997077942, 0.051180094480514526], [0.0075002312660217285, 0.25931796431541443, 0.013504356145858765, 0.38065314292907715], [0.3905479311943054, 0.20227286219596863, 0.22368675470352173, 0.24777323007583618], [0.0075002312660217285, 0.25931796431541443, 0.013504356145858765, 0.38065314292907715]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_598c3892c4b031944ab49ddfbc2bbfdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([4329], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_8556696559092b004bdcc57b7c9d29e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.01666249893605709, 0.003314714413136244]], [[0.004053822718560696, 0.0006664209067821503]], [[0.0019431465771049261, 0.003587140701711178]], [[0.029846083372831345, 0.02684077061712742]], [[0.005597985349595547, 0.0021670570131391287]], [[0.14419999718666077, 0.05131029710173607]]]], dtype='float32').reshape([1, 6, 1, 2]),
                paddle.to_tensor([-1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_be1f057b55507d5087f0dfde7b23a7a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0d7385a53f751734fdb25c5922a528b4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[0.0019043288193643093, 0.00015663662634324282]], [[0.013697666116058826, 0.001827340922318399]], [[0.0012445304309949279, 0.09536285698413849]], [[0.01113487221300602, 0.046624600887298584]], [[0.06389683485031128, 0.07109943777322769]], [[0.004716935567557812, 0.014792771078646183]]]], dtype='float32').reshape([1, 6, 1, 2]),
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


    class TestPrimitiveOp_cdb317c640df13a08100adfecb34e523(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.20323149859905243, 0.09429150074720383, 0.07140893489122391, 0.1901603490114212, 0.16385573148727417, 0.09601243585348129, 0.1287786364555359, 0.23864251375198364, 0.07157084345817566, 0.2625637948513031, 0.20802748203277588, 0.1473717987537384, 0.26525965332984924, 0.2729385793209076, 0.06359413266181946, 0.20311571657657623], dtype='float32').reshape([16]),
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


    class TestPrimitiveOp_a71620a5100a55d97ba67fb55a684baa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a71620a5100a55d97ba67fb55a684baa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1723, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_58c2367b1108f005196aa88284d925f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.016139358282089233, 0.024570047855377197, 0.1798972338438034, 0.38466477394104004], [0.3155452609062195, 0.11460816860198975, 0.05521531403064728, 0.06962801516056061], [0.045381829142570496, 0.20968413352966309, 0.11353392899036407, 0.1670221984386444], [0.3187718987464905, 0.2138129323720932, 0.054648011922836304, 0.12318618595600128], [0.14328435063362122, 0.12580473721027374, 0.005007922649383545, 0.018766164779663086]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_bc2a969e7d8bd563a490ee629613b736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.17177636921405792, 0.1657608598470688, 0.2102193832397461, 0.09437558054924011], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, 0.017763465642929077, 0.1887105405330658], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, 0.017763465642929077, 0.1887105405330658]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_7f22870d421c66797175410e6d038860(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_7f22870d421c66797175410e6d038860(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([5498, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_482840b2cc81df7024af91891968d97b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0048261284828186035, 0.07155326008796692, 0.013605579733848572, 0.051180481910705566], [0.14474661648273468, 0.2696493864059448, 0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, 0.3962211310863495, 0.15296784043312073], [0.14474661648273468, 0.2696493864059448, 0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, 0.3962211310863495, 0.15296784043312073], [0.22924979031085968, 0.1217564269900322, 0.07167693972587585, 0.07170835137367249], [0.22924979031085968, 0.1217564269900322, 0.07167693972587585, 0.07170835137367249]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_661679e2dae185a4238e1a077c3daa9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_661679e2dae185a4238e1a077c3daa9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1759, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7b25c2f5454639877b218ce75b26712a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.13257187604904175, 0.18898636102676392, 0.11631551384925842, 0.19848103821277618, 0.05323219299316406, 0.1827613115310669, 0.2413797378540039, 0.21782590448856354, 0.04551813006401062, 0.05982864275574684, 0.13052749633789062, 0.024526027962565422, 0.22515442967414856, 0.21618947386741638, 0.008748939260840416, 0.002882672706618905, 0.04029916226863861, 0.21624203026294708, 0.21809883415699005, 0.1499156802892685, 0.0032108670566231012, 0.17229510843753815, 0.06436792761087418, 0.251303493976593], dtype='float32').reshape([24]),
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


    class TestPrimitiveOp_34ae3b3722d9c4595d800392dc587cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_34ae3b3722d9c4595d800392dc587cc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1538, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_94088bf4cc3599563802769dae860c5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.08394922316074371, 0.24046628177165985, 0.1896965354681015, 0.04368181526660919], dtype='float32').reshape([4]),
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


    class TestPrimitiveOp_50ae13996dd112ce8593e5d3182be790(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.07304361462593079, 0.10842274129390717, 0.23478034138679504, 0.023416876792907715], [0.03650416433811188, 0.2645467519760132, 0.3254188299179077, 0.08861064910888672], [0.26510629057884216, 0.05608522891998291, 0.21578019857406616, 0.10049128532409668], [0.07779538631439209, 0.06618337333202362, 0.022951990365982056, 0.24318337440490723], [0.07779538631439209, 0.06618337333202362, 0.022951990365982056, 0.24318337440490723], [0.26510629057884216, 0.05608522891998291, 0.21578019857406616, 0.10049128532409668]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_a23f051432a665f7d02dc4704f41e2a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.013753771781921387, 0.2868351638317108, 0.03862294554710388, 0.19162918627262115], [0.37577491998672485, 0.05430358648300171, 0.24721841514110565, 0.08289992064237595], [0.13008549809455872, 0.14730504155158997, 0.15147680044174194, 0.33067959547042847], [0.2519821524620056, 0.363572895526886, 0.131232351064682, 0.2127522975206375], [0.013753771781921387, 0.2868351638317108, 0.03862294554710388, 0.19162918627262115]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_4a331dab29dac32c3ba5f20a95798206(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10226257890462875, 0.13534961640834808, 0.08023402094841003, 0.039990901947021484], [0.29901987314224243, 0.09267288446426392, 0.20913521945476532, 0.4559893012046814], [0.2016167938709259, 0.13266520202159882, 0.10226882994174957, 0.20165568590164185], [0.008473038673400879, 0.02449806034564972, 0.06644713133573532, 0.26101258397102356]], dtype='float32').reshape([4, 4]),
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


    class TestPrimitiveOp_74911bf8d07bd08de789cfec4a7f9f5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_74911bf8d07bd08de789cfec4a7f9f5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2135, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7a9268f41a7403ab8aeae7fa676462c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1575099527835846, 0.26694709062576294, 0.34720781445503235, 0.21742114424705505], [0.1575099527835846, 0.26694709062576294, 0.34720781445503235, 0.21742114424705505], [0.1509353071451187, 0.1447482705116272, 0.2221795916557312, 0.09455010294914246], [0.12404389679431915, 0.06935502588748932, 0.12357345223426819, 0.3634084165096283], [0.013999328017234802, 0.39129024744033813, 0.18125993013381958, 0.20690134167671204], [0.006215885281562805, 0.10090631246566772, 0.3123472034931183, 0.14281310141086578], [0.3736182451248169, 0.07743585109710693, 0.2678108513355255, 0.25866973400115967]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_26a14739b54e2110f3cd4d4c5e2bb648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_26a14739b54e2110f3cd4d4c5e2bb648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4590, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_f6642275d12646909d7869ad2bd99073(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([4851], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8f43eeae6692a23a61372ecdd306a31e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([1224], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b21b24c8001e06f3af7f18d6bff484e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b21b24c8001e06f3af7f18d6bff484e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([1042, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d598a115f9d5c06e975b0b7e43951e57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1208452433347702, 0.1525149941444397, 0.18495747447013855, 0.21681168675422668], [0.016017157584428787, 0.0064589232206344604, 0.21185481548309326, 0.15630508959293365], [0.016017157584428787, 0.0064589232206344604, 0.21185481548309326, 0.15630508959293365], [0.35850846767425537, 0.1164940744638443, 0.17222779989242554, 0.30740097165107727], [0.02179768681526184, 0.04435417056083679, 0.2214246392250061, 0.19965055584907532], [0.01850026845932007, 0.061271607875823975, 0.1330414116382599, 0.09196221828460693]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_ce21f54e7b44b2b2c11be2c68471c487(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_ce21f54e7b44b2b2c11be2c68471c487(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2339, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_a433c1e4efc66fae4394d1aa83aaed00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_a433c1e4efc66fae4394d1aa83aaed00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([3063, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d6fc1463b67746cfbc75902f16892432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_d6fc1463b67746cfbc75902f16892432(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([3822, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d7ebddabdf2f425609b19a81c7d18f62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.to_tensor([0.04458913579583168, 0.21033427119255066, 0.07286225259304047, 0.03924795612692833, 0.12533174455165863, 0.22624242305755615, 0.09203226864337921, 0.15758462250232697, 0.02457907423377037, 0.21276026964187622, 0.06444792449474335, 0.07556305080652237, 0.19715236127376556, 0.07277515530586243, 0.18578313291072845, 0.08678262680768967, 0.09069261699914932, 0.12000761181116104, 0.2193279266357422, 0.14802876114845276], dtype='float32').reshape([20]),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_8fabe249d07498680777c31bde5f0468(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9f84e9f9494c36612ae9410d8da1ce3
        def get_inputs(self):
            return [
                paddle.uniform([17475], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_b64fb0fb92d5e57df0b5ebea1dcf5b7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_b64fb0fb92d5e57df0b5ebea1dcf5b7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([2057, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2265b8f6cfbd45deca49439ba56cb17c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0021021366119384766, 0.10599559545516968, 0.1741461157798767, 0.11614590883255005], [0.2877083122730255, 0.18331629037857056, 0.15134862065315247, 0.30544114112854004], [0.3105199337005615, 0.16365931928157806, 0.19212684035301208, 0.06973066926002502], [0.3105199337005615, 0.16365931928157806, 0.19212684035301208, 0.06973066926002502], [0.2566255033016205, 0.30786266922950745, 0.02622893452644348, 0.3976331949234009]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_97535d1181c8d2805eecccbdcb052c1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([], dtype='int64'),
            ]


    class TestPrimitiveOp_97535d1181c8d2805eecccbdcb052c1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.uniform([4189, 1], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e334c18292c0d82843a7b631fd8d6b92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_256292bbbeccbfef50cde6a645d4e339
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.026198573410511017, 0.122957244515419, 0.025363638997077942, 0.051180094480514526], [0.26046913862228394, 0.16602113842964172, 0.2520030736923218, 0.0859014093875885], [0.27233195304870605, 0.3711124062538147, 0.03736155107617378, 0.2736709713935852], [0.026198573410511017, 0.122957244515419, 0.025363638997077942, 0.051180094480514526], [0.0075002312660217285, 0.25931796431541443, 0.013504356145858765, 0.38065314292907715], [0.3905479311943054, 0.20227286219596863, 0.22368675470352173, 0.24777323007583618], [0.0075002312660217285, 0.25931796431541443, 0.013504356145858765, 0.38065314292907715]], dtype='float32').reshape([7, 4]),
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