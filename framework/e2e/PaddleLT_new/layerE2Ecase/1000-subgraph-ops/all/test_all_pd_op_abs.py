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
            PADDLE_DEBUG_ENABLE_CINN=False,
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
    PADDLE_DEBUG_CINN_STAGE_NAME="backend",
    PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF=False,
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





last_stage_failed = (IsCinnStageEnableDiff() and LastCINNStageFailed())
class PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42f3e819c2d7cc4a96447324414c90a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9bd6098584f77042aca9340fb757dedc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1acec7ee93cc6ead1ac09671b0909e43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ad099c5b00eec955c8d4ab82fd7800c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_d926397ea49ebaec008021a33053e131(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ffcaa1315dd09b16d290ed78f51ce919(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96308e73d19b33181e7098009735dc51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.34457138180732727, 0.3283785581588745, 0.24062220752239227, 0.07494273036718369], [-0.3451903462409973, 0.39832618832588196, -0.11819062381982803, 0.16237980127334595], [-0.12548017501831055, 0.12842081487178802, -0.13945195078849792, -0.15617738664150238], [0.051904499530792236, 0.3438987135887146, -0.2583705484867096, -0.04573345184326172], [-0.23200926184654236, 0.05326095223426819, -0.05520813167095184, -0.013767004013061523]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_94e63b94857c4490c73a25b3adcaf01c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.14052081108093262, -0.27431637048721313, -0.316109836101532, -0.015412651002407074], [-0.32239121198654175, 0.320223867893219, -0.1386847048997879, 0.08053350448608398], [0.019224733114242554, 0.07310211658477783, 0.16810202598571777, 0.0519925057888031], [-0.32239121198654175, 0.320223867893219, -0.1386847048997879, 0.08053350448608398], [0.019224733114242554, 0.07310211658477783, 0.16810202598571777, 0.0519925057888031]], dtype='float32').reshape([5, 4]),
        ]



class PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d23068e88b59c473c1d55c166e13047(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ce92237695a03b3fd4ae66badc94dc83(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1eaf3df5e2ed81f4af37bb2fdf4b3614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17244015634059906, 0.3065081238746643, 0.19526219367980957, -0.007369339466094971], [0.051718324422836304, -0.1559659242630005, -0.14887785911560059, -0.202399343252182], [-0.21162888407707214, 0.036213815212249756, 0.18170344829559326, 0.06491860747337341], [0.051718324422836304, -0.1559659242630005, -0.14887785911560059, -0.202399343252182], [-0.21162888407707214, 0.036213815212249756, 0.18170344829559326, 0.06491860747337341], [-0.38832908868789673, -0.06375068426132202, -0.3187732696533203, 0.0030520856380462646], [-0.38832908868789673, -0.06375068426132202, -0.3187732696533203, 0.0030520856380462646]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_317505ba8c6aef334d447484b9a51c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_359d4df2bf5a9a558a2ae40222558ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fcc609a6f85619e5ae52377a70eecb5a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c9f7fda6b04beaf1cee2c44eb05445c6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1683d5a417a96737bdde6db44d304e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.310827374458313, -0.12868091464042664, -0.1052204966545105, 0.09105049818754196], [0.11764101684093475, 0.0024705082178115845, -0.3150675892829895, 0.20545414090156555], [-0.20356836915016174, 0.2083887755870819, 0.3708818256855011, -0.10785618424415588], [0.2511121332645416, -0.39549872279167175, -0.10586272180080414, -0.2821248769760132], [0.2511121332645416, -0.39549872279167175, -0.10586272180080414, -0.2821248769760132], [-0.20356836915016174, 0.2083887755870819, 0.3708818256855011, -0.10785618424415588]], dtype='float32').reshape([6, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5028521fb9d992a7536dc0ade26b7f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05068385601043701, -0.16673311591148376, -0.2300732284784317, -0.24210090935230255], [-0.35616421699523926, 0.17659874260425568, -0.23223045468330383, -0.3801487684249878], [0.26994895935058594, 0.4522857964038849, 0.1328149288892746, 0.030688166618347168], [-0.21257023513317108, 0.22678229212760925, -0.23901420831680298, 0.03767147660255432], [0.05068385601043701, -0.16673311591148376, -0.2300732284784317, -0.24210090935230255]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c3cdca0ecbd944d6052c253aa10e6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_875668963db863cdfd5005a437abe517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.004069730639457703, 0.010554656386375427, -0.15731903910636902, 0.022939056158065796], [0.0008910298347473145, -0.286990225315094, -0.21521617472171783, -0.021217264235019684], [0.15388554334640503, -0.08037617802619934, -0.04625310003757477, -0.05773010849952698], [0.3052303194999695, 0.32758742570877075, 0.1837000548839569, 0.15675829350948334]], dtype='float32').reshape([4, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29ae46566881c74026cbc33c81b657f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_3379d6fe8ff42880b1cb8b8abf50eca6(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f037189b473c6a8ff4433ef9506660da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10087436437606812, 0.18165478110313416, 0.07762834429740906, 0.1512605845928192], [0.10087436437606812, 0.18165478110313416, 0.07762834429740906, 0.1512605845928192], [0.33564990758895874, -0.3104721009731293, 0.27692002058029175, -0.03392884135246277], [0.23599711060523987, -0.06273534148931503, -0.2291894257068634, 0.02711370587348938], [0.14675763249397278, -0.01799476146697998, -0.1890169084072113, -0.03482763469219208], [-0.23367193341255188, 0.1320287436246872, -0.11780114471912384, 0.301679790019989], [-0.3079644739627838, -0.07124839723110199, 0.033957481384277344, -0.039034560322761536]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7498214574dd72d374685393fb70189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e6ec971f06fb20c7dffe0359eba1969e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_50e18615ee4e4ea9c8c4d67cd4022205(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fba3602c17073742efc8ff8fbb7eb5eb(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2e2bcc6eea8ca4ddf56450fc182f3e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2865210175514221, 0.2584184408187866, -0.04859420657157898, 0.03268370032310486], [0.044031232595443726, -0.025725483894348145, 0.11887145042419434, 0.13020625710487366], [0.044031232595443726, -0.025725483894348145, 0.11887145042419434, 0.13020625710487366], [0.32529202103614807, -0.345274418592453, 0.037326134741306305, 0.03559239208698273], [0.04636518657207489, -0.2940976917743683, 0.011127203702926636, -0.27025794982910156], [-0.3943295180797577, 0.1963176131248474, 0.07541161775588989, -0.13109560310840607]], dtype='float32').reshape([6, 4]),
        ]



class PrimitiveOp_0aabeb1a1c7f4fc3cf2c22504bc24db5(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6ccbab02660e2e6c435c37c926c03850(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_0aabeb1a1c7f4fc3cf2c22504bc24db5
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
        ]



class PrimitiveOp_a9091857dccdd48b1f987aa8f880da63(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_567bdff3912809446bd1ca945cb328a1(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a9091857dccdd48b1f987aa8f880da63
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_149512f814521e6fa5240c63cee3cd20(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_0558390dcfbfbfbf730884e49e81e0a5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_311ddb9f336c6f8a5a44bd868f2b7830(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_79b8b9a2b3a1a4ce5efa2008ec3342fd(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ccfa6811d4827c1d783ec4b0bfe950b5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c05f754a9b7102fde118c7ad7035bc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_deb84d337d6dfe17198c581d2ee653b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f1e5e21fe97dd436504b5aa30b08c2ec(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9f80e86d66e22f979bf757a27f34e3b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f094763ca7b43d3df23bff5f0564d9d1
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dff1978fc973a0ca6c8e2531584cf8ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14267420768737793, -0.03885960578918457, 0.32942402362823486, 0.07282064855098724], [0.2277335226535797, 0.006478197872638702, 0.07658667862415314, 0.06815648078918457], [-0.03666844964027405, -0.057081758975982666, -0.10836301743984222, 0.017186634242534637], [-0.03666844964027405, -0.057081758975982666, -0.10836301743984222, 0.017186634242534637], [-0.13514640927314758, 0.06575410068035126, -0.1788448691368103, -0.04929649829864502]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19a50f2ef779b6a824d8dc50973f5279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_83e37866e4aaf1f41605e8b412c7a3bf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_d926397ea49ebaec008021a33053e131
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6536440f2d7fc541669ca3e4fd304a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.30746346712112427, -0.11755669116973877, -0.0748388022184372, -0.17983797192573547], [0.1579456925392151, 0.24700625240802765, -0.1228017508983612, -0.19343301653862], [-0.007386982440948486, -0.1985030472278595, -0.13069400191307068, -0.06534376740455627], [-0.30746346712112427, -0.11755669116973877, -0.0748388022184372, -0.17983797192573547], [-0.061234280467033386, -0.28017112612724304, 0.09236519038677216, -0.04049038887023926], [0.26632726192474365, -0.0016095638275146484, 0.1833929717540741, -0.40396252274513245], [-0.061234280467033386, -0.28017112612724304, 0.09236519038677216, -0.04049038887023926]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cdf13b1808d8b2c37694594accad3bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_42f3e819c2d7cc4a96447324414c90a8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9bd6098584f77042aca9340fb757dedc(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1acec7ee93cc6ead1ac09671b0909e43(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ad099c5b00eec955c8d4ab82fd7800c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d962bb3a594276ea7c0956a95b50e9f2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1777, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_96308e73d19b33181e7098009735dc51(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.34457138180732727, 0.3283785581588745, 0.24062220752239227, 0.07494273036718369], [-0.3451903462409973, 0.39832618832588196, -0.11819062381982803, 0.16237980127334595], [-0.12548017501831055, 0.12842081487178802, -0.13945195078849792, -0.15617738664150238], [0.051904499530792236, 0.3438987135887146, -0.2583705484867096, -0.04573345184326172], [-0.23200926184654236, 0.05326095223426819, -0.05520813167095184, -0.013767004013061523]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_94e63b94857c4490c73a25b3adcaf01c(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.14052081108093262, -0.27431637048721313, -0.316109836101532, -0.015412651002407074], [-0.32239121198654175, 0.320223867893219, -0.1386847048997879, 0.08053350448608398], [0.019224733114242554, 0.07310211658477783, 0.16810202598571777, 0.0519925057888031], [-0.32239121198654175, 0.320223867893219, -0.1386847048997879, 0.08053350448608398], [0.019224733114242554, 0.07310211658477783, 0.16810202598571777, 0.0519925057888031]], dtype='float32').reshape([5, 4]),
        ]



class PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8947a642679c7ddf12a15b90d5656a86(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_ed7f493622363ba5d08d3c6897509f3e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([5480, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1eaf3df5e2ed81f4af37bb2fdf4b3614(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.17244015634059906, 0.3065081238746643, 0.19526219367980957, -0.007369339466094971], [0.051718324422836304, -0.1559659242630005, -0.14887785911560059, -0.202399343252182], [-0.21162888407707214, 0.036213815212249756, 0.18170344829559326, 0.06491860747337341], [0.051718324422836304, -0.1559659242630005, -0.14887785911560059, -0.202399343252182], [-0.21162888407707214, 0.036213815212249756, 0.18170344829559326, 0.06491860747337341], [-0.38832908868789673, -0.06375068426132202, -0.3187732696533203, 0.0030520856380462646], [-0.38832908868789673, -0.06375068426132202, -0.3187732696533203, 0.0030520856380462646]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_317505ba8c6aef334d447484b9a51c66(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_359d4df2bf5a9a558a2ae40222558ed2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7d9f67d0216e8638e405e33856d319d4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1742, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_c19f900e1e9b46356ecb59cb7c22676b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1527, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1683d5a417a96737bdde6db44d304e8f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.310827374458313, -0.12868091464042664, -0.1052204966545105, 0.09105049818754196], [0.11764101684093475, 0.0024705082178115845, -0.3150675892829895, 0.20545414090156555], [-0.20356836915016174, 0.2083887755870819, 0.3708818256855011, -0.10785618424415588], [0.2511121332645416, -0.39549872279167175, -0.10586272180080414, -0.2821248769760132], [0.2511121332645416, -0.39549872279167175, -0.10586272180080414, -0.2821248769760132], [-0.20356836915016174, 0.2083887755870819, 0.3708818256855011, -0.10785618424415588]], dtype='float32').reshape([6, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5028521fb9d992a7536dc0ade26b7f10(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.05068385601043701, -0.16673311591148376, -0.2300732284784317, -0.24210090935230255], [-0.35616421699523926, 0.17659874260425568, -0.23223045468330383, -0.3801487684249878], [0.26994895935058594, 0.4522857964038849, 0.1328149288892746, 0.030688166618347168], [-0.21257023513317108, 0.22678229212760925, -0.23901420831680298, 0.03767147660255432], [0.05068385601043701, -0.16673311591148376, -0.2300732284784317, -0.24210090935230255]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_1c3cdca0ecbd944d6052c253aa10e6de(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_875668963db863cdfd5005a437abe517(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.004069730639457703, 0.010554656386375427, -0.15731903910636902, 0.022939056158065796], [0.0008910298347473145, -0.286990225315094, -0.21521617472171783, -0.021217264235019684], [0.15388554334640503, -0.08037617802619934, -0.04625310003757477, -0.05773010849952698], [0.3052303194999695, 0.32758742570877075, 0.1837000548839569, 0.15675829350948334]], dtype='float32').reshape([4, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b29ae46566881c74026cbc33c81b657f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_634ac817e14a82c4148d4f9f92dc66e2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([2066, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f037189b473c6a8ff4433ef9506660da(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.10087436437606812, 0.18165478110313416, 0.07762834429740906, 0.1512605845928192], [0.10087436437606812, 0.18165478110313416, 0.07762834429740906, 0.1512605845928192], [0.33564990758895874, -0.3104721009731293, 0.27692002058029175, -0.03392884135246277], [0.23599711060523987, -0.06273534148931503, -0.2291894257068634, 0.02711370587348938], [0.14675763249397278, -0.01799476146697998, -0.1890169084072113, -0.03482763469219208], [-0.23367193341255188, 0.1320287436246872, -0.11780114471912384, 0.301679790019989], [-0.3079644739627838, -0.07124839723110199, 0.033957481384277344, -0.039034560322761536]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d7498214574dd72d374685393fb70189(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d0fd2624cb61b35704a9373b7d3aaeda(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_8c42eec98da05bb34bb4a4a3a621e363(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([4586, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_41a2cdf8521c6810c5aa0ba8e40ad31a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([1073, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2e2bcc6eea8ca4ddf56450fc182f3e89(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.2865210175514221, 0.2584184408187866, -0.04859420657157898, 0.03268370032310486], [0.044031232595443726, -0.025725483894348145, 0.11887145042419434, 0.13020625710487366], [0.044031232595443726, -0.025725483894348145, 0.11887145042419434, 0.13020625710487366], [0.32529202103614807, -0.345274418592453, 0.037326134741306305, 0.03559239208698273], [0.04636518657207489, -0.2940976917743683, 0.011127203702926636, -0.27025794982910156], [-0.3943295180797577, 0.1963176131248474, 0.07541161775588989, -0.13109560310840607]], dtype='float32').reshape([6, 4]),
        ]



class PrimitiveOp_140781b5a63852c07669f83ad482e5b9(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0):
        input_0 = arg_0
        return paddle._C_ops.abs(input_0)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_77939128bfdb23a71cc043004251dde3(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_140781b5a63852c07669f83ad482e5b9
    def get_inputs(self):
        return [
            paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_e0f4e278d6cfc0e59815bbcca73f0bbf(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_140781b5a63852c07669f83ad482e5b9
    def get_inputs(self):
        return [
            paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4315fa6a82073fa03ecc8b7a1d28f962(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_f44d258b5fce527af6ecc0cf20053c73(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([2383, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_07824636053be0f12be85c45508a5d2f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([3030, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_a2191136f49489c0011d18aa4e016f1a(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([3787, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_45b3fe7313c9f328a2f7430ed9fef981(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7c05f754a9b7102fde118c7ad7035bc0(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_deb84d337d6dfe17198c581d2ee653b2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_d0b9cbfe521f37395a1b5d032251fcd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([2084, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_862ad5c08c96fe27bda5e83be53b59b7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_a4ff3fffb2f87a97e46dcd79001ac905
    def get_inputs(self):
        return [
            paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dff1978fc973a0ca6c8e2531584cf8ee(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.14267420768737793, -0.03885960578918457, 0.32942402362823486, 0.07282064855098724], [0.2277335226535797, 0.006478197872638702, 0.07658667862415314, 0.06815648078918457], [-0.03666844964027405, -0.057081758975982666, -0.10836301743984222, 0.017186634242534637], [-0.03666844964027405, -0.057081758975982666, -0.10836301743984222, 0.017186634242534637], [-0.13514640927314758, 0.06575410068035126, -0.1788448691368103, -0.04929649829864502]], dtype='float32').reshape([5, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19a50f2ef779b6a824d8dc50973f5279(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7086bb359d13df44389ec7a1179b7498(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([4260, 4], dtype='float32', min=0, max=0.5),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6536440f2d7fc541669ca3e4fd304a6f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[-0.30746346712112427, -0.11755669116973877, -0.0748388022184372, -0.17983797192573547], [0.1579456925392151, 0.24700625240802765, -0.1228017508983612, -0.19343301653862], [-0.007386982440948486, -0.1985030472278595, -0.13069400191307068, -0.06534376740455627], [-0.30746346712112427, -0.11755669116973877, -0.0748388022184372, -0.17983797192573547], [-0.061234280467033386, -0.28017112612724304, 0.09236519038677216, -0.04049038887023926], [0.26632726192474365, -0.0016095638275146484, 0.1833929717540741, -0.40396252274513245], [-0.061234280467033386, -0.28017112612724304, 0.09236519038677216, -0.04049038887023926]], dtype='float32').reshape([7, 4]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_cdf13b1808d8b2c37694594accad3bb5(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_77c8249ab2c6791f53f330aae30aaa0f
    def get_inputs(self):
        return [
            paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
        ]




if __name__ == '__main__':
    unittest.main()