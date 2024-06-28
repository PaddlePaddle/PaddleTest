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
    class PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19dce77a46cbb426fcb175872eef85f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0285a47cd53f28f1bca1dc757dbb0ee2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3be7cdfdaf034170159940941a59e6cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05312109af1ba1be64851486030a30c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4e289836100e50c55308b13873350467(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f58337eb630bf5cc7efdedc57a29bdf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f51ac2d47b1d7771c2fb7f80d5fa981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.059835709631443024, 0.2431793510913849, -0.06668603420257568, 0.09330688416957855], [-0.030676662921905518, 0.4153245687484741, -0.09938794374465942, 0.10088853538036346], [-0.041919320821762085, 0.11580070853233337, 0.13541805744171143, -0.10899001359939575], [0.19998770952224731, -0.3514717221260071, 0.011652082204818726, -0.3247314393520355], [0.10408538579940796, 0.18593794107437134, 0.10364016890525818, 0.07083243876695633]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_c804cbc6a260b6722484add35a3bb625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.07945215702056885, 0.19198602437973022, -0.09651744365692139, -0.051396384835243225], [-0.1376815140247345, -0.03616875410079956, -0.030297398567199707, 0.08579754829406738], [0.06492412090301514, -0.02363431453704834, 0.03141921013593674, -0.15331938862800598], [-0.1376815140247345, -0.03616875410079956, -0.030297398567199707, 0.08579754829406738], [0.06492412090301514, -0.02363431453704834, 0.03141921013593674, -0.15331938862800598]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_c64be4311dbf84c7b1014b537b8202cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba2332ce94aef1e5607844c363f8682c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64be4311dbf84c7b1014b537b8202cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36b089d8e635dfdf2c1f5ce7c44ea210(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93014d5c813824139a347abe2352b525(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10889467597007751, 0.25807124376296997, -0.14413365721702576, 0.4194449782371521], [0.3009010851383209, -0.18025171756744385, 0.11982408165931702, -0.061148613691329956], [0.36769217252731323, 0.1304820477962494, -0.4372129440307617, 0.006102010607719421], [0.3009010851383209, -0.18025171756744385, 0.11982408165931702, -0.061148613691329956], [0.36769217252731323, 0.1304820477962494, -0.4372129440307617, 0.006102010607719421], [0.1060771495103836, -0.28093963861465454, -0.11857184767723083, -0.316922664642334], [0.1060771495103836, -0.28093963861465454, -0.11857184767723083, -0.316922664642334]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_c5d96756fb9d6c4d0800cf7302a18cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23877c6eb8a3260b16f5afc2bfc94541(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3d157441bbef240a470f425ec960d239(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b82edbe7969c901eda5d96349aa3f5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a35cdf76dbcc09eea2bf32b29a2cc69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22957313060760498, -0.17837467789649963, 0.016854897141456604, -0.40951916575431824], [0.030538499355316162, -0.026057451963424683, 0.00427737832069397, -0.09727845340967178], [-0.25486868619918823, -0.3133251667022705, -0.23783250153064728, -0.25276681780815125], [0.15342719852924347, -0.05401526391506195, -0.19605214893817902, -0.08178121596574783], [0.15342719852924347, -0.05401526391506195, -0.19605214893817902, -0.08178121596574783], [-0.25486868619918823, -0.3133251667022705, -0.23783250153064728, -0.25276681780815125]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_3bf287fec888ead4efa797ef34492695(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21438781917095184, 0.07429340481758118, -0.2580089867115021, -0.2164541482925415], [-0.13806931674480438, -0.42606988549232483, -0.00037895888090133667, 0.46145597100257874], [0.12322920560836792, 0.12052872776985168, 0.28811296820640564, -0.20321156084537506], [0.05649891495704651, 0.06143069267272949, 0.13873469829559326, -0.31833985447883606], [0.21438781917095184, 0.07429340481758118, -0.2580089867115021, -0.2164541482925415]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_a071f6c6f1d23110a6910c304befcb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0ec7d57b2b215e22418a453d5a8a95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04451256990432739, -0.19473513960838318, -0.36136090755462646, 0.024676114320755005], [-0.04359281063079834, 0.4449837803840637, 0.11287054419517517, 0.006923645734786987], [0.2663431763648987, 0.1137508675456047, 0.3312119245529175, 0.14764076471328735], [-0.16015861928462982, 0.13828155398368835, 0.009508371353149414, -0.2052965611219406]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_a836513d26728b27e45ec1c529b625b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4521165d9ad8b044a2ebd1060985e2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fe0017cee95acc0d792185514ee5085(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03216975927352905, 0.04337479919195175, -0.01657208800315857, -0.09818089008331299], [0.03216975927352905, 0.04337479919195175, -0.01657208800315857, -0.09818089008331299], [0.15343743562698364, -0.3510548174381256, -0.23534007370471954, -0.0949157178401947], [0.12012717127799988, -0.08196210116147995, -0.07514175772666931, 0.4337972104549408], [0.061709702014923096, -0.21900923550128937, 0.12456387281417847, -0.1701676845550537], [-0.083442822098732, 0.27063196897506714, 0.05398473143577576, 0.12532399594783783], [-0.09118098020553589, 0.2385154515504837, -0.08096741139888763, -0.00832115113735199]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_441575f7dc1e437ba993dd1640d4f01e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56af57ef6f6baa1bbf0204edc0ebb096(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64be4311dbf84c7b1014b537b8202cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d36423b7db83fe7dc7a8048c41effb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_537a85c79126500ab1ed6bbee999f77a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8da4f1be5f91f81ea9f70fe838e0c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.17413032054901123, -0.02193164825439453, 0.06767009198665619, -0.10391233116388321], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [-0.08004461228847504, -0.23684172332286835, -0.0887954980134964, -0.17910811305046082], [0.05183491110801697, 0.12347441911697388, 0.13930287957191467, 0.2521556615829468], [-0.21063363552093506, 0.13145971298217773, -0.15847629308700562, 0.02333889901638031]], dtype='float32').reshape([6, 4]),
            ]


    
    class PrimitiveOp_f2a672b80ca67edc24b95c771929f1ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7115b3484e55b2b225703bb464fdf75e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2a672b80ca67edc24b95c771929f1ed
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f3d7b9b0746d006b676f0f79bf5bf518(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d471028d4e3e24eb46d7095b4b07643f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f3d7b9b0746d006b676f0f79bf5bf518
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62c9f12f1f32a37f59c7bf464d99c825(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64be4311dbf84c7b1014b537b8202cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9aedbe996a359f0dcb36efdea80ea2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1f218e278a4a6093bf22d5641357d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2da5cd7ef864d196843388555c50358(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe32869320ebecb79414eab828004150(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64be4311dbf84c7b1014b537b8202cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a986e57cf5c4dfeaacba02f18d9b02f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a7385f01fe3a866a11b66b919da8a56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca1004aecfacf04ca2d8189227bab318(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef632f895ffb8f973fa88f8338c0445e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64be4311dbf84c7b1014b537b8202cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f6f83fd7b92ec157ed550afea04bc98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36759066581726074, 0.1356358379125595, 0.07934285700321198, -0.07276766747236252], [-0.37631040811538696, -0.03991866111755371, -0.0015145167708396912, -0.07648283243179321], [-0.07664056122303009, 0.06367561221122742, 0.027359262108802795, -0.13421794772148132], [-0.07664056122303009, 0.06367561221122742, 0.027359262108802795, -0.13421794772148132], [0.2624303996562958, -0.038322463631629944, 0.20256070792675018, -0.42651233077049255]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_4fc8c070e10b8257efade3d932abe398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ae2eab5fd8505106e0734a402a1ce6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09d200e43edadfcd94d5abf717b852d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.15342693030834198, 0.16269968450069427, -0.16442686319351196, 0.3595154881477356], [-0.0890268087387085, 0.00014069676399230957, 0.25438398122787476, 0.21876737475395203], [-0.28361520171165466, 0.2432878464460373, -0.06699240207672119, 0.010794661939144135], [-0.15342693030834198, 0.16269968450069427, -0.16442686319351196, 0.3595154881477356], [0.3432466685771942, 0.06492901593446732, -0.25622114539146423, -0.2037186324596405], [-0.3985275328159332, 0.0876234769821167, 0.07240909337997437, 0.14636847376823425], [0.3432466685771942, 0.06492901593446732, -0.25622114539146423, -0.2037186324596405]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_87b2d8f1d9d0516bbf25830d29350384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b539ac3b8b364a601990fae1bc125e21(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1024, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c1e7ada49b768f184551ba72b4ebe22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b539ac3b8b364a601990fae1bc125e21
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_114706bafbe0b765c4411554d90fb0a6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4096, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_109d7f3152b8fe3df9664df651c2bcd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_114706bafbe0b765c4411554d90fb0a6
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_605a7bfe9aa796ee459b358266562bf7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8e7f0d9fce549448bd0a915b857f70d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_605a7bfe9aa796ee459b358266562bf7
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7a035d4a1ca6dbeb4d4fb25b4cc97de4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[53, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ed46b2ee0b455cea4892344288a34ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a035d4a1ca6dbeb4d4fb25b4cc97de4
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0fa84b0e269c9f0ea4f97427de8de41(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1774, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_61183fc6c7b9b2691205a2f97f2cfe4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0fa84b0e269c9f0ea4f97427de8de41
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b03206884fb2651827882967eb6d09e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1fa20db2cfc4dd7d5c6217b7f985ac33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.059835709631443024, 0.2431793510913849, -0.06668603420257568, 0.09330688416957855], [-0.030676662921905518, 0.4153245687484741, -0.09938794374465942, 0.10088853538036346], [-0.041919320821762085, 0.11580070853233337, 0.13541805744171143, -0.10899001359939575], [0.19998770952224731, -0.3514717221260071, 0.011652082204818726, -0.3247314393520355], [0.10408538579940796, 0.18593794107437134, 0.10364016890525818, 0.07083243876695633]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_cebfbfe905d0e71f06b398d6790cf6e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.07945215702056885, 0.19198602437973022, -0.09651744365692139, -0.051396384835243225], [-0.1376815140247345, -0.03616875410079956, -0.030297398567199707, 0.08579754829406738], [0.06492412090301514, -0.02363431453704834, 0.03141921013593674, -0.15331938862800598], [-0.1376815140247345, -0.03616875410079956, -0.030297398567199707, 0.08579754829406738], [0.06492412090301514, -0.02363431453704834, 0.03141921013593674, -0.15331938862800598]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_5d56b8657685644394fc5d0cf81682b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 8, 8], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f301d04fe2eaa7b03b391a9042033607(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d56b8657685644394fc5d0cf81682b2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_272c8956bb663fbd81036927e78489c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5454, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_19fed4f422d18802cf32555c0be8ae13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_272c8956bb663fbd81036927e78489c3
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8e9fa1482a123f77e934f7d6d11509ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10889467597007751, 0.25807124376296997, -0.14413365721702576, 0.4194449782371521], [0.3009010851383209, -0.18025171756744385, 0.11982408165931702, -0.061148613691329956], [0.36769217252731323, 0.1304820477962494, -0.4372129440307617, 0.006102010607719421], [0.3009010851383209, -0.18025171756744385, 0.11982408165931702, -0.061148613691329956], [0.36769217252731323, 0.1304820477962494, -0.4372129440307617, 0.006102010607719421], [0.1060771495103836, -0.28093963861465454, -0.11857184767723083, -0.316922664642334], [0.1060771495103836, -0.28093963861465454, -0.11857184767723083, -0.316922664642334]], dtype='float32').reshape([7, 4]),
            ]


    
    class PrimitiveOp_d4b80da2d1a5b68a0263ad3e67544b75(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_65cb50d712a9089a1445fb71ddc0e672(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4b80da2d1a5b68a0263ad3e67544b75
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f9c446c0339e60a3bdd78ecaa1daf45(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[103, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3261669afcdd6a675fdd1ad6ae9c62d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f9c446c0339e60a3bdd78ecaa1daf45
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd98a4ecd4668a12ea77c3ae6fe9eb76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1722, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad3c64985058fb5efb23003f96f2d729(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd98a4ecd4668a12ea77c3ae6fe9eb76
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4d4020fb39b3ce3c31bc8aef01284b41(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1518, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_da123ef94fe196772ae485996fc9c548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d4020fb39b3ce3c31bc8aef01284b41
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_11acaa3e3b1883649c3a05188c9dd575(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_27c28f1144c114dfcd24fd23f2a5ba94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acaa3e3b1883649c3a05188c9dd575
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22957313060760498, -0.17837467789649963, 0.016854897141456604, -0.40951916575431824], [0.030538499355316162, -0.026057451963424683, 0.00427737832069397, -0.09727845340967178], [-0.25486868619918823, -0.3133251667022705, -0.23783250153064728, -0.25276681780815125], [0.15342719852924347, -0.05401526391506195, -0.19605214893817902, -0.08178121596574783], [0.15342719852924347, -0.05401526391506195, -0.19605214893817902, -0.08178121596574783], [-0.25486868619918823, -0.3133251667022705, -0.23783250153064728, -0.25276681780815125]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_e2b9b3e7fd996c497fd138cc25443253(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21438781917095184, 0.07429340481758118, -0.2580089867115021, -0.2164541482925415], [-0.13806931674480438, -0.42606988549232483, -0.00037895888090133667, 0.46145597100257874], [0.12322920560836792, 0.12052872776985168, 0.28811296820640564, -0.20321156084537506], [0.05649891495704651, 0.06143069267272949, 0.13873469829559326, -0.31833985447883606], [0.21438781917095184, 0.07429340481758118, -0.2580089867115021, -0.2164541482925415]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_5a902b8cdc6510a8d898bb436ff317b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45f34b072596c1111d5c980437c69e60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a902b8cdc6510a8d898bb436ff317b7
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b23a2f4d39a9de5445545c7a96497782(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f3d17c34973473e5832845a65b01df2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b23a2f4d39a9de5445545c7a96497782
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04451256990432739, -0.19473513960838318, -0.36136090755462646, 0.024676114320755005], [-0.04359281063079834, 0.4449837803840637, 0.11287054419517517, 0.006923645734786987], [0.2663431763648987, 0.1137508675456047, 0.3312119245529175, 0.14764076471328735], [-0.16015861928462982, 0.13828155398368835, 0.009508371353149414, -0.2052965611219406]], dtype='float32').reshape([4, 4]),
            ]


    
    class PrimitiveOp_75d1824c51526b44890a13c1e1a29ba2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[84, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96a86fc61bd3794466314bf436542eb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_75d1824c51526b44890a13c1e1a29ba2
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5df3f0523c81600cbbe4ea712843dd32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2133, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf441643272e22ba3607158ad6ee3b9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5df3f0523c81600cbbe4ea712843dd32
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87894ec7cf950c5f377aa0600d2ec6bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03216975927352905, 0.04337479919195175, -0.01657208800315857, -0.09818089008331299], [0.03216975927352905, 0.04337479919195175, -0.01657208800315857, -0.09818089008331299], [0.15343743562698364, -0.3510548174381256, -0.23534007370471954, -0.0949157178401947], [0.12012717127799988, -0.08196210116147995, -0.07514175772666931, 0.4337972104549408], [0.061709702014923096, -0.21900923550128937, 0.12456387281417847, -0.1701676845550537], [-0.083442822098732, 0.27063196897506714, 0.05398473143577576, 0.12532399594783783], [-0.09118098020553589, 0.2385154515504837, -0.08096741139888763, -0.00832115113735199]], dtype='float32').reshape([7, 4]),
            ]


    
    class PrimitiveOp_e9a672407c91abe6d192f5ceab7f0f69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16384, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f49ed9c58fb7e715e195304a62180b3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9a672407c91abe6d192f5ceab7f0f69
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0233d6dc11f997ff00a4aee32ba4899c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 64, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac720a4bb89183d2039b9bd44b4a0c0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0233d6dc11f997ff00a4aee32ba4899c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_faa26ade9e0a7038994e1e6c13e113b6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4631, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0bc6a4b4808d1f955bf0535ede56341b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_faa26ade9e0a7038994e1e6c13e113b6
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9042413ec3a93d7e0a76b6aa90c18415(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1039, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cfa474e8b929cf7a3636f9c16b38bf5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9042413ec3a93d7e0a76b6aa90c18415
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_28d9b861d5b477d4edfd1a15fc8e46f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acaa3e3b1883649c3a05188c9dd575
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.17413032054901123, -0.02193164825439453, 0.06767009198665619, -0.10391233116388321], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [-0.08004461228847504, -0.23684172332286835, -0.0887954980134964, -0.17910811305046082], [0.05183491110801697, 0.12347441911697388, 0.13930287957191467, 0.2521556615829468], [-0.21063363552093506, 0.13145971298217773, -0.15847629308700562, 0.02333889901638031]], dtype='float32').reshape([6, 4]),
            ]


    
    class PrimitiveOp_4715d7b1d9fdc6e3a2f31d7a58f10d8b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 2, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1bbb8968c7ba748a3055ff5c7a2ca7e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4715d7b1d9fdc6e3a2f31d7a58f10d8b
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3bab87825f586c3a16c465638b7e3a00(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[300, 2, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4159ba2eef4539182f2c3463809a43a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bab87825f586c3a16c465638b7e3a00
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fd39b09a7d9f78cb2999e36642a484b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 128, 128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e86523403913436c194824f8b5bd2c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd39b09a7d9f78cb2999e36642a484b2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_32aba7fff9ed4399ec0843d7d66e602c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2318, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b8ebb556109eee3b82ab41d90236790b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32aba7fff9ed4399ec0843d7d66e602c
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b14f8633d9bc0f198ae5fb72de256ca0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2961, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51ddc467f7f6c9ecea634db322eab938(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b14f8633d9bc0f198ae5fb72de256ca0
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa8cabac1cda67e661ec5c35f1b64cc0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3739, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45701a1914911ca4ae0baba59c7b40b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa8cabac1cda67e661ec5c35f1b64cc0
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f37184cce3da011dfad3b69db16cada(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 16, 16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a81d85134634640e4561d7b3330dd0c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f37184cce3da011dfad3b69db16cada
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dbaf81305c077852ef683ee7453d6b68(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[256, 5], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_05bcef9183895d51205f9c294ff8c4f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbaf81305c077852ef683ee7453d6b68
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_267e1ab30da0de101a77876de2ac2a57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[47, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_56457b41454b81738de11d3694f92919(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_267e1ab30da0de101a77876de2ac2a57
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ebf34b0f71e7114b81aeb7a21d59543b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2013, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_886fe933baa11881874e1c8d15872d68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ebf34b0f71e7114b81aeb7a21d59543b
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c855722a153a47146dfaf628f978d9b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 32, 32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8e005206477aa278cbed5f0499da7650(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c855722a153a47146dfaf628f978d9b0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_daafeb04df0353e013dcb005a2fdaf12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36759066581726074, 0.1356358379125595, 0.07934285700321198, -0.07276766747236252], [-0.37631040811538696, -0.03991866111755371, -0.0015145167708396912, -0.07648283243179321], [-0.07664056122303009, 0.06367561221122742, 0.027359262108802795, -0.13421794772148132], [-0.07664056122303009, 0.06367561221122742, 0.027359262108802795, -0.13421794772148132], [0.2624303996562958, -0.038322463631629944, 0.20256070792675018, -0.42651233077049255]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_e317e6181aef35269d786be3d1bf5d63(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0f6ef3944a177b5dfad1454430430b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e317e6181aef35269d786be3d1bf5d63
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_72136e46c6678432f1abd75bd0178af0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4177, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62dbe07ca8aa193d0c63863044240ddb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72136e46c6678432f1abd75bd0178af0
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6aed8a1120b3028109af4542505f8e53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.15342693030834198, 0.16269968450069427, -0.16442686319351196, 0.3595154881477356], [-0.0890268087387085, 0.00014069676399230957, 0.25438398122787476, 0.21876737475395203], [-0.28361520171165466, 0.2432878464460373, -0.06699240207672119, 0.010794661939144135], [-0.15342693030834198, 0.16269968450069427, -0.16442686319351196, 0.3595154881477356], [0.3432466685771942, 0.06492901593446732, -0.25622114539146423, -0.2037186324596405], [-0.3985275328159332, 0.0876234769821167, 0.07240909337997437, 0.14636847376823425], [0.3432466685771942, 0.06492901593446732, -0.25622114539146423, -0.2037186324596405]], dtype='float32').reshape([7, 4]),
            ]


    
    class PrimitiveOp_28d603187b17a4ed18c569ae2e14cbc9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[52, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_623915dd86d10bbd60addc5ea79d5279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_28d603187b17a4ed18c569ae2e14cbc9
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19dce77a46cbb426fcb175872eef85f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1024, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0285a47cd53f28f1bca1dc757dbb0ee2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([4096, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3be7cdfdaf034170159940941a59e6cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05312109af1ba1be64851486030a30c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([53, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e42dd298d0b55899a2c347ceccd13e94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1774, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f51ac2d47b1d7771c2fb7f80d5fa981(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.059835709631443024, 0.2431793510913849, -0.06668603420257568, 0.09330688416957855], [-0.030676662921905518, 0.4153245687484741, -0.09938794374465942, 0.10088853538036346], [-0.041919320821762085, 0.11580070853233337, 0.13541805744171143, -0.10899001359939575], [0.19998770952224731, -0.3514717221260071, 0.011652082204818726, -0.3247314393520355], [0.10408538579940796, 0.18593794107437134, 0.10364016890525818, 0.07083243876695633]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_c804cbc6a260b6722484add35a3bb625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.07945215702056885, 0.19198602437973022, -0.09651744365692139, -0.051396384835243225], [-0.1376815140247345, -0.03616875410079956, -0.030297398567199707, 0.08579754829406738], [0.06492412090301514, -0.02363431453704834, 0.03141921013593674, -0.15331938862800598], [-0.1376815140247345, -0.03616875410079956, -0.030297398567199707, 0.08579754829406738], [0.06492412090301514, -0.02363431453704834, 0.03141921013593674, -0.15331938862800598]], dtype='float32').reshape([5, 4]),
            ]


    
    class PrimitiveOp_eaef0a1f4c8873a7876078fa4185f29a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ad5b8c1dbecc529fa9bc36c3dee5c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaef0a1f4c8873a7876078fa4185f29a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_822a43ca17c1b4ba04b91df7853b2cfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([5454, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_93014d5c813824139a347abe2352b525(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10889467597007751, 0.25807124376296997, -0.14413365721702576, 0.4194449782371521], [0.3009010851383209, -0.18025171756744385, 0.11982408165931702, -0.061148613691329956], [0.36769217252731323, 0.1304820477962494, -0.4372129440307617, 0.006102010607719421], [0.3009010851383209, -0.18025171756744385, 0.11982408165931702, -0.061148613691329956], [0.36769217252731323, 0.1304820477962494, -0.4372129440307617, 0.006102010607719421], [0.1060771495103836, -0.28093963861465454, -0.11857184767723083, -0.316922664642334], [0.1060771495103836, -0.28093963861465454, -0.11857184767723083, -0.316922664642334]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_c5d96756fb9d6c4d0800cf7302a18cbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([64, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23877c6eb8a3260b16f5afc2bfc94541(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([103, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92f31df5e8ccd78e6c07578f147db031(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1722, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e5032f217fa6bc8ff972d6b9f875b0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1518, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a35cdf76dbcc09eea2bf32b29a2cc69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.22957313060760498, -0.17837467789649963, 0.016854897141456604, -0.40951916575431824], [0.030538499355316162, -0.026057451963424683, 0.00427737832069397, -0.09727845340967178], [-0.25486868619918823, -0.3133251667022705, -0.23783250153064728, -0.25276681780815125], [0.15342719852924347, -0.05401526391506195, -0.19605214893817902, -0.08178121596574783], [0.15342719852924347, -0.05401526391506195, -0.19605214893817902, -0.08178121596574783], [-0.25486868619918823, -0.3133251667022705, -0.23783250153064728, -0.25276681780815125]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_3bf287fec888ead4efa797ef34492695(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.21438781917095184, 0.07429340481758118, -0.2580089867115021, -0.2164541482925415], [-0.13806931674480438, -0.42606988549232483, -0.00037895888090133667, 0.46145597100257874], [0.12322920560836792, 0.12052872776985168, 0.28811296820640564, -0.20321156084537506], [0.05649891495704651, 0.06143069267272949, 0.13873469829559326, -0.31833985447883606], [0.21438781917095184, 0.07429340481758118, -0.2580089867115021, -0.2164541482925415]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_a071f6c6f1d23110a6910c304befcb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a0ec7d57b2b215e22418a453d5a8a95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.04451256990432739, -0.19473513960838318, -0.36136090755462646, 0.024676114320755005], [-0.04359281063079834, 0.4449837803840637, 0.11287054419517517, 0.006923645734786987], [0.2663431763648987, 0.1137508675456047, 0.3312119245529175, 0.14764076471328735], [-0.16015861928462982, 0.13828155398368835, 0.009508371353149414, -0.2052965611219406]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_a836513d26728b27e45ec1c529b625b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_354ce9e348071cab4a92d55748533eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2133, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fe0017cee95acc0d792185514ee5085(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.03216975927352905, 0.04337479919195175, -0.01657208800315857, -0.09818089008331299], [0.03216975927352905, 0.04337479919195175, -0.01657208800315857, -0.09818089008331299], [0.15343743562698364, -0.3510548174381256, -0.23534007370471954, -0.0949157178401947], [0.12012717127799988, -0.08196210116147995, -0.07514175772666931, 0.4337972104549408], [0.061709702014923096, -0.21900923550128937, 0.12456387281417847, -0.1701676845550537], [-0.083442822098732, 0.27063196897506714, 0.05398473143577576, 0.12532399594783783], [-0.09118098020553589, 0.2385154515504837, -0.08096741139888763, -0.00832115113735199]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_441575f7dc1e437ba993dd1640d4f01e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([16384, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_166569aede17259e8010d933c83f8ea1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaef0a1f4c8873a7876078fa4185f29a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8a375bb9d47b3c5159d62d83ceae1e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([4631, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2363c82bd418cea21671d71f8ae47a8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8da4f1be5f91f81ea9f70fe838e0c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.17413032054901123, -0.02193164825439453, 0.06767009198665619, -0.10391233116388321], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [0.3498445749282837, 0.04219639301300049, 0.021489620208740234, 0.18335233628749847], [-0.08004461228847504, -0.23684172332286835, -0.0887954980134964, -0.17910811305046082], [0.05183491110801697, 0.12347441911697388, 0.13930287957191467, 0.2521556615829468], [-0.21063363552093506, 0.13145971298217773, -0.15847629308700562, 0.02333889901638031]], dtype='float32').reshape([6, 4]),
            ]


    
    class PrimitiveOp_280ab8a772240a28478ca8f74a5e924b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8fad117bd61bf7722b1447f5d0e7df0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_280ab8a772240a28478ca8f74a5e924b
        def get_inputs(self):
            return [
                paddle.uniform([100, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_227f57726e47cf53efafe096da9a9662(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_280ab8a772240a28478ca8f74a5e924b
        def get_inputs(self):
            return [
                paddle.uniform([300, 2, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a4e63079fa7a42d80d1f8781c848658(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaef0a1f4c8873a7876078fa4185f29a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c19e8d9c19b482c4692d800da814ad67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2318, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f4865298a49036d4c4a6947cdf7420f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2961, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_161c0918a4dc14bffcbc1dc95ec7d54b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([3739, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c9114d17404592e47b42308046c4e33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaef0a1f4c8873a7876078fa4185f29a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a986e57cf5c4dfeaacba02f18d9b02f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([256, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a7385f01fe3a866a11b66b919da8a56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([47, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e7405d7b1d73e5661ca1bf20fdfa171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2013, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19f9833b15d9ffcc31b28175421ba45f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaef0a1f4c8873a7876078fa4185f29a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f6f83fd7b92ec157ed550afea04bc98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.36759066581726074, 0.1356358379125595, 0.07934285700321198, -0.07276766747236252], [-0.37631040811538696, -0.03991866111755371, -0.0015145167708396912, -0.07648283243179321], [-0.07664056122303009, 0.06367561221122742, 0.027359262108802795, -0.13421794772148132], [-0.07664056122303009, 0.06367561221122742, 0.027359262108802795, -0.13421794772148132], [0.2624303996562958, -0.038322463631629944, 0.20256070792675018, -0.42651233077049255]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_4fc8c070e10b8257efade3d932abe398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf3263bf50d638b05a8b71e8759f6f27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([4177, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09d200e43edadfcd94d5abf717b852d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.15342693030834198, 0.16269968450069427, -0.16442686319351196, 0.3595154881477356], [-0.0890268087387085, 0.00014069676399230957, 0.25438398122787476, 0.21876737475395203], [-0.28361520171165466, 0.2432878464460373, -0.06699240207672119, 0.010794661939144135], [-0.15342693030834198, 0.16269968450069427, -0.16442686319351196, 0.3595154881477356], [0.3432466685771942, 0.06492901593446732, -0.25622114539146423, -0.2037186324596405], [-0.3985275328159332, 0.0876234769821167, 0.07240909337997437, 0.14636847376823425], [0.3432466685771942, 0.06492901593446732, -0.25622114539146423, -0.2037186324596405]], dtype='float32').reshape([7, 4]),
            ]


    class TestPrimitiveOp_87b2d8f1d9d0516bbf25830d29350384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([52, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()