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
class PrimitiveOp_e85bfea412a45e69a2429e66ba612040(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.distribute_fpn_proposals(input_0, input_1, 2, 5, 4, 224, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[1], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_918b5ecf972cb774138deb0930ffec29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e85bfea412a45e69a2429e66ba612040
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300], dtype='int32').reshape([1]),
        ]



class PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.distribute_fpn_proposals(input_0, input_1, 2, 5, 4, 224, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[None], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78633d0590e59435f8c5fee448a55bc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]



class PrimitiveOp_3c69b3e68b0f21b2e410831504204dc7(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.distribute_fpn_proposals(input_0, input_1, 2, 5, 4, 224, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[512, 4], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dfe33dd6efa7792c3445257c3eab26d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c69b3e68b0f21b2e410831504204dc7
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b9081f3ef3499945391ad6f5f97aac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e85bfea412a45e69a2429e66ba612040
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6be0d975f337714db158b0f1817b473b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dfe33dd6efa7792c3445257c3eab26d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c69b3e68b0f21b2e410831504204dc7
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78633d0590e59435f8c5fee448a55bc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538935fcca85531e1b29b76aa477e2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9178119620369f4ad266696a817c28ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0744657814502716, 0.12276715785264969, 0.35531410574913025, 0.48441699147224426], [0.3495769500732422, 0.41702592372894287, 0.4514731168746948, 0.1187334656715393], [0.08253531157970428, 0.12003565579652786, 0.47612228989601135, 0.23035909235477448], [0.2246844619512558, 0.05810641497373581, 0.18594786524772644, 0.02734232135117054], [0.0647694543004036, 0.20631584525108337, 0.4361419379711151, 0.3645848035812378], [0.18269386887550354, 0.03176182508468628, 0.43640458583831787, 0.15800030529499054]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([6], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fb4ca99478afd2448a8533871efab973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_dfe33dd6efa7792c3445257c3eab26d2(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_3c69b3e68b0f21b2e410831504204dc7
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9473cd993e81be2ed09bf3fb4d3ac575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4721968472003937, 0.46351638436317444, 0.43057215213775635, 0.4704304039478302], [0.28280454874038696, 0.06356373429298401, 0.09064285457134247, 0.47968965768814087]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71ae25c71902ba96b2cbb167cf559666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([390, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e27cc283555b77f706cd7d513785f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13273146748542786, 0.17586477100849152, 0.22157639265060425, 0.061416372656822205], [0.3036189675331116, 0.09734106063842773, 0.1676512509584427, 0.3660214841365814]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_218915f958d1cb11a063c001ccc548aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538935fcca85531e1b29b76aa477e2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538935fcca85531e1b29b76aa477e2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19fe6e6c46e2b636587bba0e75c940ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40654340386390686, 0.14714907109737396, 0.07348904013633728, 0.17775322496891022], [0.3003535270690918, 0.396862655878067, 0.44543522596359253, 0.02731742151081562]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_918b5ecf972cb774138deb0930ffec29(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e85bfea412a45e69a2429e66ba612040
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_218915f958d1cb11a063c001ccc548aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_53df0aed7e997c57a924b56d5bf996ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04137223934f5bec1b7eb4e626433d4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.258802205324173, 0.4104510545730591, 0.17734280228614807, 0.4375358819961548], [0.2670711278915405, 0.33529600501060486, 0.4748607575893402, 0.3120517432689667]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4d9912be1ad85fcdc7966fb041f6c59e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37470003962516785, 0.09868413209915161, 0.11757244914770126, 0.29152926802635193], [0.4315420389175415, 0.13176584243774414, 0.4546758234500885, 0.10029768943786621], [0.4329819679260254, 0.12452353537082672, 0.05320322886109352, 0.22823214530944824], [0.04202219471335411, 0.36871975660324097, 0.3148946464061737, 0.04706772789359093]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_5b9081f3ef3499945391ad6f5f97aac9(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_e85bfea412a45e69a2429e66ba612040
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ade46637c30b6c16c7a74deb2835042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4508846402168274, 0.3680139482021332, 0.10928935557603836, 0.2164817601442337], [0.014404275454580784, 0.4741973280906677, 0.12449511885643005, 0.26038020849227905]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538935fcca85531e1b29b76aa477e2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21ff484003bc22613bc6ad53cd53d978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4429657757282257, 0.47139474749565125, 0.27933281660079956, 0.44871142506599426], [0.3590521216392517, 0.3183683753013611, 0.10797044634819031, 0.43915653228759766], [0.1250833421945572, 0.309425413608551, 0.11987120658159256, 0.11405402421951294]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([6], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b2d3fae4556d8935438833a0112db67f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78633d0590e59435f8c5fee448a55bc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]



class PrimitiveOp_106a68f46f5281f8e889705f5c9ed015(InstanceTrait, paddle.nn.Layer):
    
    def __init__(self):
        super().__init__()

    def forward(self, arg_0, arg_1):
        input_0 = arg_0
        input_1 = arg_1
        return paddle._C_ops.distribute_fpn_proposals(input_0, input_1, 2, 5, 4, 224, False)

    def get_input_spec(self):
        return [
            paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            paddle.static.InputSpec(shape=[], dtype='int32'),
        ]
        
    instance_ = None
    static_instance_with_cinn_ = None
    static_instance_without_cinn_ = None


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bbc17bb9bc61b56662311b149f8252a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_106a68f46f5281f8e889705f5c9ed015
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bedbbd34e1cdefd1920839b3c2f2ecd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_6be0d975f337714db158b0f1817b473b(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([2], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bbc17bb9bc61b56662311b149f8252a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_106a68f46f5281f8e889705f5c9ed015
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_78633d0590e59435f8c5fee448a55bc7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538935fcca85531e1b29b76aa477e2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9178119620369f4ad266696a817c28ea(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.0744657814502716, 0.12276715785264969, 0.35531410574913025, 0.48441699147224426], [0.3495769500732422, 0.41702592372894287, 0.4514731168746948, 0.1187334656715393], [0.08253531157970428, 0.12003565579652786, 0.47612228989601135, 0.23035909235477448], [0.2246844619512558, 0.05810641497373581, 0.18594786524772644, 0.02734232135117054], [0.0647694543004036, 0.20631584525108337, 0.4361419379711151, 0.3645848035812378], [0.18269386887550354, 0.03176182508468628, 0.43640458583831787, 0.15800030529499054]], dtype='float32').reshape([6, 4]),
            paddle.to_tensor([6], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_fb4ca99478afd2448a8533871efab973(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([3], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bbc17bb9bc61b56662311b149f8252a7(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_106a68f46f5281f8e889705f5c9ed015
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor(2, dtype='int32').reshape([]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_9473cd993e81be2ed09bf3fb4d3ac575(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4721968472003937, 0.46351638436317444, 0.43057215213775635, 0.4704304039478302], [0.28280454874038696, 0.06356373429298401, 0.09064285457134247, 0.47968965768814087]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_71ae25c71902ba96b2cbb167cf559666(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([390, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_7e27cc283555b77f706cd7d513785f69(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.13273146748542786, 0.17586477100849152, 0.22157639265060425, 0.061416372656822205], [0.3036189675331116, 0.09734106063842773, 0.1676512509584427, 0.3660214841365814]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_218915f958d1cb11a063c001ccc548aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538935fcca85531e1b29b76aa477e2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538935fcca85531e1b29b76aa477e2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_19fe6e6c46e2b636587bba0e75c940ad(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.40654340386390686, 0.14714907109737396, 0.07348904013633728, 0.17775322496891022], [0.3003535270690918, 0.396862655878067, 0.44543522596359253, 0.02731742151081562]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_b2d3fae4556d8935438833a0112db67f(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([300, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([300], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_218915f958d1cb11a063c001ccc548aa(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([5], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_53df0aed7e997c57a924b56d5bf996ac(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_04137223934f5bec1b7eb4e626433d4e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.258802205324173, 0.4104510545730591, 0.17734280228614807, 0.4375358819961548], [0.2670711278915405, 0.33529600501060486, 0.4748607575893402, 0.3120517432689667]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_4d9912be1ad85fcdc7966fb041f6c59e(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.37470003962516785, 0.09868413209915161, 0.11757244914770126, 0.29152926802635193], [0.4315420389175415, 0.13176584243774414, 0.4546758234500885, 0.10029768943786621], [0.4329819679260254, 0.12452353537082672, 0.05320322886109352, 0.22823214530944824], [0.04202219471335411, 0.36871975660324097, 0.3148946464061737, 0.04706772789359093]], dtype='float32').reshape([4, 4]),
            paddle.to_tensor([8], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_bedbbd34e1cdefd1920839b3c2f2ecd4(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([100, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([100], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_2ade46637c30b6c16c7a74deb2835042(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4508846402168274, 0.3680139482021332, 0.10928935557603836, 0.2164817601442337], [0.014404275454580784, 0.4741973280906677, 0.12449511885643005, 0.26038020849227905]], dtype='float32').reshape([2, 4]),
            paddle.to_tensor([1], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_538935fcca85531e1b29b76aa477e2f8(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.uniform([512, 4], dtype='float32', min=0, max=0.5),
            paddle.to_tensor([7], dtype='int32').reshape([1]),
        ]


@unittest.skipIf(last_stage_failed, "last stage failed")
class TestPrimitiveOp_21ff484003bc22613bc6ad53cd53d978(CinnTestBase, unittest.TestCase):
    
    def get_test_class(self):
        return PrimitiveOp_f0169b0ad62ad9cb1c017698958aff0f
    def get_inputs(self):
        return [
            paddle.to_tensor([[0.4429657757282257, 0.47139474749565125, 0.27933281660079956, 0.44871142506599426], [0.3590521216392517, 0.3183683753013611, 0.10797044634819031, 0.43915653228759766], [0.1250833421945572, 0.309425413608551, 0.11987120658159256, 0.11405402421951294]], dtype='float32').reshape([3, 4]),
            paddle.to_tensor([6], dtype='int32').reshape([1]),
        ]




if __name__ == '__main__':
    unittest.main()