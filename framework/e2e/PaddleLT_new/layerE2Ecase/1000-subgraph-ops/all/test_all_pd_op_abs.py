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


    class TestPrimitiveOp_7e24510aead0cca9e797a462384f09bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7150c7da78cccbf21725bfc958367f42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.011946693062782288, -0.22909292578697205, -0.15888167917728424, -0.015180230140686035], [0.23457637429237366, -0.1387505978345871, 0.23476874828338623, 0.039133816957473755], [0.3682340085506439, -0.2067251205444336, -0.04031139612197876, 0.008929014205932617], [-0.2650768756866455, -0.06502872705459595, 0.3271848261356354, -0.06669780611991882], [-0.0899767279624939, 0.10820017755031586, -0.05213071405887604, -0.06857089698314667]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_ddf0530cedfc18804ea406d9b67197bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2601255178451538, -0.2813035547733307, -0.3010902404785156, -0.11394605040550232], [0.1276143491268158, -0.1411561220884323, 0.2907564043998718, -0.004007205367088318], [-0.3767981231212616, -0.019813083112239838, -0.08903086185455322, -0.06486682593822479], [0.1276143491268158, -0.1411561220884323, 0.2907564043998718, -0.004007205367088318], [-0.3767981231212616, -0.019813083112239838, -0.08903086185455322, -0.06486682593822479]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_e45021e1702961794e960187a56a245f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3cc8da8a1615c77c6a140879667f5d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3336668610572815, 0.3895747661590576, 0.027006596326828003, -0.17651008069515228], [0.2637220025062561, -0.02544151246547699, 0.1283944547176361, -0.13914698362350464], [-0.04708457738161087, 0.0057485103607177734, -0.3778526186943054, 0.16070452332496643], [0.2637220025062561, -0.02544151246547699, 0.1283944547176361, -0.13914698362350464], [-0.04708457738161087, 0.0057485103607177734, -0.3778526186943054, 0.16070452332496643], [0.06639361381530762, 0.2864929735660553, -0.26026010513305664, -0.3718429505825043], [0.06639361381530762, 0.2864929735660553, -0.26026010513305664, -0.3718429505825043]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_1f14487526fecb8bab4d872b09b80dd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8b0c9cc93f811e77534e54e29f042c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1767605b8070a190fd8aea371a71921d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.1509297639131546, -0.1552731692790985, -0.056120842695236206, 0.15356037020683289], [-0.07212063670158386, -0.3970109820365906, 0.12442433834075928, 0.38769468665122986], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079], [-0.12647047638893127, -0.19000458717346191, -0.3279547691345215, -0.06161805987358093], [-0.12647047638893127, -0.19000458717346191, -0.3279547691345215, -0.06161805987358093], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_efd11cd194c6c71e946fe6d2f7e8bef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3127082586288452, -0.048425883054733276, -0.1581532061100006, 0.0485401451587677], [-0.3431030213832855, 0.06575411558151245, -0.0661439597606659, 0.27380648255348206], [-0.06341108679771423, 0.11628088355064392, 0.10398587584495544, -0.028025232255458832], [-0.17723333835601807, 0.2572104036808014, -0.20969152450561523, 0.18656745553016663], [0.3127082586288452, -0.048425883054733276, -0.1581532061100006, 0.0485401451587677]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_a071f6c6f1d23110a6910c304befcb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9068ebe7ad70ed94ceca08cfffb5c1ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.14037862420082092, -0.3072340786457062, -0.233082577586174, -0.22903814911842346], [0.14534536004066467, -0.06035545468330383, -0.3983846604824066, -0.18683511018753052], [-0.041945427656173706, -0.24708035588264465, -0.2164594829082489, -0.12575536966323853], [-0.003619551658630371, 0.1542983204126358, -0.1320928931236267, 0.09877075254917145]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_a836513d26728b27e45ec1c529b625b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6816356d3ba371aec318b6e32ca894ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_804de21b250c7894456ae9313fa231c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.17318011820316315, 0.0040985941886901855, -0.10722425580024719, -0.24594347178936005], [-0.17318011820316315, 0.0040985941886901855, -0.10722425580024719, -0.24594347178936005], [0.2724611163139343, 0.14107611775398254, 0.3538140058517456, 0.029049724340438843], [0.0411318838596344, 0.2469097077846527, -0.05615696310997009, 0.09188510477542877], [-0.09708136320114136, -0.41709190607070923, 0.1974916011095047, 0.15154391527175903], [-0.022501900792121887, 0.2542913854122162, 0.010012298822402954, 0.1757974624633789], [0.01827526092529297, 0.11807702481746674, -0.04941102862358093, 0.02130529098212719]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_1ba4db83f65797b059927b16e96ee3f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8da05ef79e9bcd465687948ac359e36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_386a9ed029d5cea591faf68da02e5a82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08719142526388168, -0.40703436732292175, 0.14703017473220825, -0.20497220754623413], [0.21442238986492157, 0.18900448083877563, -0.22183440625667572, 0.010784268379211426], [0.21442238986492157, 0.18900448083877563, -0.22183440625667572, 0.010784268379211426], [0.2041129320859909, -0.1454058587551117, 0.39465102553367615, 0.02922683209180832], [-0.02670140564441681, 0.1606736183166504, -0.18615491688251495, -0.40934592485427856], [-0.3336673974990845, 0.08125244081020355, -0.2736116051673889, 0.33980047702789307]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_8e915ec127241b4e1a21d6d7ce29492e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f775087b5e0bccd1b36d5868916076e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1adde9410b5c43dd7645b33942ae0ac8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_4d886b614cd2ab37b3f1122aa296800b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef632f895ffb8f973fa88f8338c0445e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64be4311dbf84c7b1014b537b8202cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf3e21a1ec8d305668b7db857b908519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.05997924134135246, 0.17256204783916473, 0.29755768179893494, 0.03335516154766083], [0.43779996037483215, 0.19319099187850952, 0.01612231135368347, 0.18234547972679138], [-0.17438164353370667, 0.24513502418994904, -0.027111470699310303, 0.13506805896759033], [-0.17438164353370667, 0.24513502418994904, -0.027111470699310303, 0.13506805896759033], [0.13616040349006653, -0.334640234708786, -0.23136883974075317, -0.01873648166656494]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_4fc8c070e10b8257efade3d932abe398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4cc42b7b107e273ea0bb89491c7306c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34fb8bda2bc0044081ccc855dd7d54c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3046402335166931, -0.07770112156867981, 0.14432917535305023, 0.180344358086586], [-0.03918623924255371, 0.019960127770900726, -0.028510063886642456, -0.033728450536727905], [-0.017332687973976135, -0.28235897421836853, 0.3564165532588959, -0.1784440129995346], [0.3046402335166931, -0.07770112156867981, 0.14432917535305023, 0.180344358086586], [0.055790454149246216, -0.36793333292007446, -0.027703553438186646, 0.24950526654720306], [0.23903867602348328, -0.20549030601978302, 0.09378381073474884, -0.2969999313354492], [0.055790454149246216, -0.36793333292007446, -0.027703553438186646, 0.24950526654720306]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_18f405d20f1742264939b91fa6781b8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1696, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9430b4bd9c30760a4eae528ddc210657(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18f405d20f1742264939b91fa6781b8f
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_273509940235ef07e281ca47e87ba1c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.011946693062782288, -0.22909292578697205, -0.15888167917728424, -0.015180230140686035], [0.23457637429237366, -0.1387505978345871, 0.23476874828338623, 0.039133816957473755], [0.3682340085506439, -0.2067251205444336, -0.04031139612197876, 0.008929014205932617], [-0.2650768756866455, -0.06502872705459595, 0.3271848261356354, -0.06669780611991882], [-0.0899767279624939, 0.10820017755031586, -0.05213071405887604, -0.06857089698314667]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_7c1d74f561a63d8dbc07e0c6909fbdc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2601255178451538, -0.2813035547733307, -0.3010902404785156, -0.11394605040550232], [0.1276143491268158, -0.1411561220884323, 0.2907564043998718, -0.004007205367088318], [-0.3767981231212616, -0.019813083112239838, -0.08903086185455322, -0.06486682593822479], [0.1276143491268158, -0.1411561220884323, 0.2907564043998718, -0.004007205367088318], [-0.3767981231212616, -0.019813083112239838, -0.08903086185455322, -0.06486682593822479]], dtype='float32').reshape([5, 4]),
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


    
    class PrimitiveOp_2b50c153f03edc903a47389dcc9117e6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5517, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c645d5f471349bf0cbeed64a780028b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b50c153f03edc903a47389dcc9117e6
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_7c7ed77581eb5db4b76df8f28ab3f9ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3336668610572815, 0.3895747661590576, 0.027006596326828003, -0.17651008069515228], [0.2637220025062561, -0.02544151246547699, 0.1283944547176361, -0.13914698362350464], [-0.04708457738161087, 0.0057485103607177734, -0.3778526186943054, 0.16070452332496643], [0.2637220025062561, -0.02544151246547699, 0.1283944547176361, -0.13914698362350464], [-0.04708457738161087, 0.0057485103607177734, -0.3778526186943054, 0.16070452332496643], [0.06639361381530762, 0.2864929735660553, -0.26026010513305664, -0.3718429505825043], [0.06639361381530762, 0.2864929735660553, -0.26026010513305664, -0.3718429505825043]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_5780f4c1129df4d76a7db45bfb166677(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1794, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5b487071dd9f95bb1d756db1fb29b6af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5780f4c1129df4d76a7db45bfb166677
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8befe8388d541716942f8301a9e2b613(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1504, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30a8a24a97ecc41f2e09c774203e499b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8befe8388d541716942f8301a9e2b613
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_fe7e180ed26eed3bbfa2a5d986de0aaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acaa3e3b1883649c3a05188c9dd575
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.1509297639131546, -0.1552731692790985, -0.056120842695236206, 0.15356037020683289], [-0.07212063670158386, -0.3970109820365906, 0.12442433834075928, 0.38769468665122986], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079], [-0.12647047638893127, -0.19000458717346191, -0.3279547691345215, -0.06161805987358093], [-0.12647047638893127, -0.19000458717346191, -0.3279547691345215, -0.06161805987358093], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_567802f88f3d2b9358d4f372b93057f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3127082586288452, -0.048425883054733276, -0.1581532061100006, 0.0485401451587677], [-0.3431030213832855, 0.06575411558151245, -0.0661439597606659, 0.27380648255348206], [-0.06341108679771423, 0.11628088355064392, 0.10398587584495544, -0.028025232255458832], [-0.17723333835601807, 0.2572104036808014, -0.20969152450561523, 0.18656745553016663], [0.3127082586288452, -0.048425883054733276, -0.1581532061100006, 0.0485401451587677]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_efbab25ac928f0ad9bd93045362b49c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b23a2f4d39a9de5445545c7a96497782
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.14037862420082092, -0.3072340786457062, -0.233082577586174, -0.22903814911842346], [0.14534536004066467, -0.06035545468330383, -0.3983846604824066, -0.18683511018753052], [-0.041945427656173706, -0.24708035588264465, -0.2164594829082489, -0.12575536966323853], [-0.003619551658630371, 0.1542983204126358, -0.1320928931236267, 0.09877075254917145]], dtype='float32').reshape([4, 4]),
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


    
    class PrimitiveOp_fb44a9fb88cc02520c8db8a1655414c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2039, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c7c21959aa98717fb2dffd6a7b906ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb44a9fb88cc02520c8db8a1655414c8
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e22e9b1b51480af88e2c8e6835172c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.17318011820316315, 0.0040985941886901855, -0.10722425580024719, -0.24594347178936005], [-0.17318011820316315, 0.0040985941886901855, -0.10722425580024719, -0.24594347178936005], [0.2724611163139343, 0.14107611775398254, 0.3538140058517456, 0.029049724340438843], [0.0411318838596344, 0.2469097077846527, -0.05615696310997009, 0.09188510477542877], [-0.09708136320114136, -0.41709190607070923, 0.1974916011095047, 0.15154391527175903], [-0.022501900792121887, 0.2542913854122162, 0.010012298822402954, 0.1757974624633789], [0.01827526092529297, 0.11807702481746674, -0.04941102862358093, 0.02130529098212719]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_dc42704f67247d10f0b3c5435cd80520(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4584, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fa4118bb9f55053d45e06d4593da83df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc42704f67247d10f0b3c5435cd80520
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a3d1c7cd57e276de15720b3924c626c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1071, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d5b4a9cc66a388739d73e9a302af772(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a3d1c7cd57e276de15720b3924c626c
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a27c326248c13427037784d29c748edf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acaa3e3b1883649c3a05188c9dd575
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08719142526388168, -0.40703436732292175, 0.14703017473220825, -0.20497220754623413], [0.21442238986492157, 0.18900448083877563, -0.22183440625667572, 0.010784268379211426], [0.21442238986492157, 0.18900448083877563, -0.22183440625667572, 0.010784268379211426], [0.2041129320859909, -0.1454058587551117, 0.39465102553367615, 0.02922683209180832], [-0.02670140564441681, 0.1606736183166504, -0.18615491688251495, -0.40934592485427856], [-0.3336673974990845, 0.08125244081020355, -0.2736116051673889, 0.33980047702789307]], dtype='float32').reshape([6, 4]),
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


    
    class PrimitiveOp_e652b62d83ca04e8246f51afe8b386bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2370, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7eefac92bd4ccd8cd4fcad9b26968f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e652b62d83ca04e8246f51afe8b386bd
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3658db525a9b094137acc81d3df68970(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2993, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c38d48dd02193052a1d847478cde07f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3658db525a9b094137acc81d3df68970
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1dae981959579563297d098f32fd7990(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3832, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c361292377a6d6147ede23142676dca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1dae981959579563297d098f32fd7990
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_eb08f619990ce483cda1ca1f01b0c7e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1995, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e887168662b738bbad0f597ca71267da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb08f619990ce483cda1ca1f01b0c7e5
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_2a44351e0248e311b847fd55b740dfb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.05997924134135246, 0.17256204783916473, 0.29755768179893494, 0.03335516154766083], [0.43779996037483215, 0.19319099187850952, 0.01612231135368347, 0.18234547972679138], [-0.17438164353370667, 0.24513502418994904, -0.027111470699310303, 0.13506805896759033], [-0.17438164353370667, 0.24513502418994904, -0.027111470699310303, 0.13506805896759033], [0.13616040349006653, -0.334640234708786, -0.23136883974075317, -0.01873648166656494]], dtype='float32').reshape([5, 4]),
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


    
    class PrimitiveOp_eb03542210d5bd9500f7f8b63d8f173c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4181, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ee80d7627372428c8963b3c1a641e433(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb03542210d5bd9500f7f8b63d8f173c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bd408b2ecf1fbaab7cc036c821456af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3046402335166931, -0.07770112156867981, 0.14432917535305023, 0.180344358086586], [-0.03918623924255371, 0.019960127770900726, -0.028510063886642456, -0.033728450536727905], [-0.017332687973976135, -0.28235897421836853, 0.3564165532588959, -0.1784440129995346], [0.3046402335166931, -0.07770112156867981, 0.14432917535305023, 0.180344358086586], [0.055790454149246216, -0.36793333292007446, -0.027703553438186646, 0.24950526654720306], [0.23903867602348328, -0.20549030601978302, 0.09378381073474884, -0.2969999313354492], [0.055790454149246216, -0.36793333292007446, -0.027703553438186646, 0.24950526654720306]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_4ec3df2b4b93a8af9a8fe712fe71ccf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1696, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7150c7da78cccbf21725bfc958367f42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.011946693062782288, -0.22909292578697205, -0.15888167917728424, -0.015180230140686035], [0.23457637429237366, -0.1387505978345871, 0.23476874828338623, 0.039133816957473755], [0.3682340085506439, -0.2067251205444336, -0.04031139612197876, 0.008929014205932617], [-0.2650768756866455, -0.06502872705459595, 0.3271848261356354, -0.06669780611991882], [-0.0899767279624939, 0.10820017755031586, -0.05213071405887604, -0.06857089698314667]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_ddf0530cedfc18804ea406d9b67197bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.2601255178451538, -0.2813035547733307, -0.3010902404785156, -0.11394605040550232], [0.1276143491268158, -0.1411561220884323, 0.2907564043998718, -0.004007205367088318], [-0.3767981231212616, -0.019813083112239838, -0.08903086185455322, -0.06486682593822479], [0.1276143491268158, -0.1411561220884323, 0.2907564043998718, -0.004007205367088318], [-0.3767981231212616, -0.019813083112239838, -0.08903086185455322, -0.06486682593822479]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_412024155bd8493e34b990afe1f37032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([5517, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3cc8da8a1615c77c6a140879667f5d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3336668610572815, 0.3895747661590576, 0.027006596326828003, -0.17651008069515228], [0.2637220025062561, -0.02544151246547699, 0.1283944547176361, -0.13914698362350464], [-0.04708457738161087, 0.0057485103607177734, -0.3778526186943054, 0.16070452332496643], [0.2637220025062561, -0.02544151246547699, 0.1283944547176361, -0.13914698362350464], [-0.04708457738161087, 0.0057485103607177734, -0.3778526186943054, 0.16070452332496643], [0.06639361381530762, 0.2864929735660553, -0.26026010513305664, -0.3718429505825043], [0.06639361381530762, 0.2864929735660553, -0.26026010513305664, -0.3718429505825043]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_cc9097fd5b67a83681a62db317b6baf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1794, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dee458b321d59e1867786ba3c8d2c47d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1767605b8070a190fd8aea371a71921d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.1509297639131546, -0.1552731692790985, -0.056120842695236206, 0.15356037020683289], [-0.07212063670158386, -0.3970109820365906, 0.12442433834075928, 0.38769468665122986], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079], [-0.12647047638893127, -0.19000458717346191, -0.3279547691345215, -0.06161805987358093], [-0.12647047638893127, -0.19000458717346191, -0.3279547691345215, -0.06161805987358093], [0.2050803303718567, 0.1458708643913269, 0.05275455117225647, 0.09451538324356079]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_efd11cd194c6c71e946fe6d2f7e8bef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3127082586288452, -0.048425883054733276, -0.1581532061100006, 0.0485401451587677], [-0.3431030213832855, 0.06575411558151245, -0.0661439597606659, 0.27380648255348206], [-0.06341108679771423, 0.11628088355064392, 0.10398587584495544, -0.028025232255458832], [-0.17723333835601807, 0.2572104036808014, -0.20969152450561523, 0.18656745553016663], [0.3127082586288452, -0.048425883054733276, -0.1581532061100006, 0.0485401451587677]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_a071f6c6f1d23110a6910c304befcb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9068ebe7ad70ed94ceca08cfffb5c1ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.14037862420082092, -0.3072340786457062, -0.233082577586174, -0.22903814911842346], [0.14534536004066467, -0.06035545468330383, -0.3983846604824066, -0.18683511018753052], [-0.041945427656173706, -0.24708035588264465, -0.2164594829082489, -0.12575536966323853], [-0.003619551658630371, 0.1542983204126358, -0.1320928931236267, 0.09877075254917145]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_a836513d26728b27e45ec1c529b625b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_088d8d4f78c9f192c1777cc0183c1f3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2039, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_804de21b250c7894456ae9313fa231c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.17318011820316315, 0.0040985941886901855, -0.10722425580024719, -0.24594347178936005], [-0.17318011820316315, 0.0040985941886901855, -0.10722425580024719, -0.24594347178936005], [0.2724611163139343, 0.14107611775398254, 0.3538140058517456, 0.029049724340438843], [0.0411318838596344, 0.2469097077846527, -0.05615696310997009, 0.09188510477542877], [-0.09708136320114136, -0.41709190607070923, 0.1974916011095047, 0.15154391527175903], [-0.022501900792121887, 0.2542913854122162, 0.010012298822402954, 0.1757974624633789], [0.01827526092529297, 0.11807702481746674, -0.04941102862358093, 0.02130529098212719]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_9e64cd2a0d7ce812f2098b259248464f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([4584, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3993bb66749eadd2602b33e3c97f43e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1071, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_386a9ed029d5cea591faf68da02e5a82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.08719142526388168, -0.40703436732292175, 0.14703017473220825, -0.20497220754623413], [0.21442238986492157, 0.18900448083877563, -0.22183440625667572, 0.010784268379211426], [0.21442238986492157, 0.18900448083877563, -0.22183440625667572, 0.010784268379211426], [0.2041129320859909, -0.1454058587551117, 0.39465102553367615, 0.02922683209180832], [-0.02670140564441681, 0.1606736183166504, -0.18615491688251495, -0.40934592485427856], [-0.3336673974990845, 0.08125244081020355, -0.2736116051673889, 0.33980047702789307]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_47674dd179a10e4210028e8803cec906(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2370, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f84e49ba6816cbecf777c13bbbc0197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2993, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_780eae2bafe51b1858aac8f1f6dd1f97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([3832, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_dec8eda102f2f8caa14d1bbe701f53b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1995, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19f9833b15d9ffcc31b28175421ba45f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaef0a1f4c8873a7876078fa4185f29a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf3e21a1ec8d305668b7db857b908519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.05997924134135246, 0.17256204783916473, 0.29755768179893494, 0.03335516154766083], [0.43779996037483215, 0.19319099187850952, 0.01612231135368347, 0.18234547972679138], [-0.17438164353370667, 0.24513502418994904, -0.027111470699310303, 0.13506805896759033], [-0.17438164353370667, 0.24513502418994904, -0.027111470699310303, 0.13506805896759033], [0.13616040349006653, -0.334640234708786, -0.23136883974075317, -0.01873648166656494]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_4fc8c070e10b8257efade3d932abe398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ee4f7cd98e56d4094f9a99bb3d90beb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([4181, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34fb8bda2bc0044081ccc855dd7d54c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3046402335166931, -0.07770112156867981, 0.14432917535305023, 0.180344358086586], [-0.03918623924255371, 0.019960127770900726, -0.028510063886642456, -0.033728450536727905], [-0.017332687973976135, -0.28235897421836853, 0.3564165532588959, -0.1784440129995346], [0.3046402335166931, -0.07770112156867981, 0.14432917535305023, 0.180344358086586], [0.055790454149246216, -0.36793333292007446, -0.027703553438186646, 0.24950526654720306], [0.23903867602348328, -0.20549030601978302, 0.09378381073474884, -0.2969999313354492], [0.055790454149246216, -0.36793333292007446, -0.027703553438186646, 0.24950526654720306]], dtype='float32').reshape([7, 4]),
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