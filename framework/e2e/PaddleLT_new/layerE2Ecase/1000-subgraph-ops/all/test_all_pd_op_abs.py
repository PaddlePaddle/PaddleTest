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


    class TestPrimitiveOp_c2c33c40f1dcdba818ee5027a9303be0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94b9d1fdce4614421df4621de7c1a289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.016139358282089233, 0.024570047855377197, 0.1798972338438034, 0.38466477394104004], [0.3155452609062195, -0.11460816860198975, 0.05521531403064728, -0.06962801516056061], [-0.045381829142570496, -0.20968413352966309, -0.11353392899036407, 0.1670221984386444], [0.3187718987464905, 0.2138129323720932, 0.054648011922836304, -0.12318618595600128], [0.14328435063362122, 0.12580473721027374, 0.005007922649383545, 0.018766164779663086]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_704c1dea06bbdffc1d60a06eac66a84f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.17177636921405792, -0.1657608598470688, 0.2102193832397461, -0.09437558054924011], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, -0.017763465642929077, -0.1887105405330658], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, -0.017763465642929077, -0.1887105405330658]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_1e2edeb044a02c4a210da9c6fb40894e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51633851c21d969c82b18f276f0aa9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0048261284828186035, -0.07155326008796692, 0.013605579733848572, -0.051180481910705566], [0.14474661648273468, 0.2696493864059448, -0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, -0.3962211310863495, -0.15296784043312073], [0.14474661648273468, 0.2696493864059448, -0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, -0.3962211310863495, -0.15296784043312073], [0.22924979031085968, 0.1217564269900322, -0.07167693972587585, -0.07170835137367249], [0.22924979031085968, 0.1217564269900322, -0.07167693972587585, -0.07170835137367249]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_4ef7b877af1c6722ed1f803b74c4b502(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ccd51d5d776b14dc088a8167cb9d4cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4809a8ae9ea3cb37c96edf5024fa6414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.07304361462593079, -0.10842274129390717, 0.23478034138679504, 0.023416876792907715], [0.03650416433811188, -0.2645467519760132, -0.3254188299179077, -0.08861064910888672], [0.26510629057884216, 0.05608522891998291, -0.21578019857406616, 0.10049128532409668], [0.07779538631439209, -0.06618337333202362, -0.022951990365982056, -0.24318337440490723], [0.07779538631439209, -0.06618337333202362, -0.022951990365982056, -0.24318337440490723], [0.26510629057884216, 0.05608522891998291, -0.21578019857406616, 0.10049128532409668]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_17c0383b6a6e7ccb0504d8f2a7c06be7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.013753771781921387, -0.2868351638317108, 0.03862294554710388, 0.19162918627262115], [-0.37577491998672485, -0.05430358648300171, 0.24721841514110565, -0.08289992064237595], [0.13008549809455872, -0.14730504155158997, 0.15147680044174194, -0.33067959547042847], [-0.2519821524620056, 0.363572895526886, -0.131232351064682, 0.2127522975206375], [0.013753771781921387, -0.2868351638317108, 0.03862294554710388, 0.19162918627262115]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_a071f6c6f1d23110a6910c304befcb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1f431dcb50d4357a191377029e43c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10226257890462875, 0.13534961640834808, -0.08023402094841003, 0.039990901947021484], [0.29901987314224243, -0.09267288446426392, -0.20913521945476532, 0.4559893012046814], [-0.2016167938709259, -0.13266520202159882, -0.10226882994174957, 0.20165568590164185], [0.008473038673400879, 0.02449806034564972, -0.06644713133573532, 0.26101258397102356]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_a836513d26728b27e45ec1c529b625b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2ca03a70e5b8b66cff7f552dfa9bf66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7ea2d3ffcdc8dd5d5c9504c6e8eaeab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1575099527835846, -0.26694709062576294, -0.34720781445503235, 0.21742114424705505], [0.1575099527835846, -0.26694709062576294, -0.34720781445503235, 0.21742114424705505], [0.1509353071451187, -0.1447482705116272, -0.2221795916557312, 0.09455010294914246], [0.12404389679431915, -0.06935502588748932, -0.12357345223426819, -0.3634084165096283], [0.013999328017234802, -0.39129024744033813, -0.18125993013381958, 0.20690134167671204], [-0.006215885281562805, -0.10090631246566772, 0.3123472034931183, 0.14281310141086578], [-0.3736182451248169, 0.07743585109710693, 0.2678108513355255, 0.25866973400115967]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_748f5c327616301c82d7df2712f9d4a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21bb2a62dd24e2149f6a9430916d4072(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72d39991db8df65da65e851dd8a198dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1208452433347702, 0.1525149941444397, -0.18495747447013855, -0.21681168675422668], [0.016017157584428787, -0.0064589232206344604, -0.21185481548309326, -0.15630508959293365], [0.016017157584428787, -0.0064589232206344604, -0.21185481548309326, -0.15630508959293365], [0.35850846767425537, 0.1164940744638443, 0.17222779989242554, -0.30740097165107727], [0.02179768681526184, 0.04435417056083679, 0.2214246392250061, -0.19965055584907532], [0.01850026845932007, 0.061271607875823975, -0.1330414116382599, 0.09196221828460693]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_687f862938bdfa852ae98675461cff44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec21cbccb61fad029ba8ff2ee1f6cae1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efb723e4b96838a8be489962336009ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_ac2c2e6f6804261053d67ca7b46f30ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef632f895ffb8f973fa88f8338c0445e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c64be4311dbf84c7b1014b537b8202cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f54a3285752cd8d40743bb3bed6deb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0021021366119384766, 0.10599559545516968, 0.1741461157798767, 0.11614590883255005], [-0.2877083122730255, 0.18331629037857056, -0.15134862065315247, 0.30544114112854004], [0.3105199337005615, -0.16365931928157806, 0.19212684035301208, -0.06973066926002502], [0.3105199337005615, -0.16365931928157806, 0.19212684035301208, -0.06973066926002502], [0.2566255033016205, 0.30786266922950745, -0.02622893452644348, -0.3976331949234009]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_4fc8c070e10b8257efade3d932abe398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cddb31e25e14fdf6ffa363cc655eb9bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e289836100e50c55308b13873350467
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d1a21df6e7fd9c9b79d90e1bcfb1b21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.026198573410511017, 0.122957244515419, 0.025363638997077942, -0.051180094480514526], [0.26046913862228394, 0.16602113842964172, 0.2520030736923218, -0.0859014093875885], [0.27233195304870605, -0.3711124062538147, -0.03736155107617378, 0.2736709713935852], [-0.026198573410511017, 0.122957244515419, 0.025363638997077942, -0.051180094480514526], [0.0075002312660217285, 0.25931796431541443, -0.013504356145858765, 0.38065314292907715], [-0.3905479311943054, -0.20227286219596863, 0.22368675470352173, 0.24777323007583618], [0.0075002312660217285, 0.25931796431541443, -0.013504356145858765, 0.38065314292907715]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_6f50e5e8aa83ce2c45c23fd5fdcec585(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1723, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f963b5a02f9f6b8b897b474d1d17287a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f50e5e8aa83ce2c45c23fd5fdcec585
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_f015d407a22ff28e8211f8f76938e8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.016139358282089233, 0.024570047855377197, 0.1798972338438034, 0.38466477394104004], [0.3155452609062195, -0.11460816860198975, 0.05521531403064728, -0.06962801516056061], [-0.045381829142570496, -0.20968413352966309, -0.11353392899036407, 0.1670221984386444], [0.3187718987464905, 0.2138129323720932, 0.054648011922836304, -0.12318618595600128], [0.14328435063362122, 0.12580473721027374, 0.005007922649383545, 0.018766164779663086]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_66217993b059dbb7417da337adf678f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.17177636921405792, -0.1657608598470688, 0.2102193832397461, -0.09437558054924011], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, -0.017763465642929077, -0.1887105405330658], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, -0.017763465642929077, -0.1887105405330658]], dtype='float32').reshape([5, 4]),
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


    
    class PrimitiveOp_bf6b4653f28b9908d3a1b94fe0008c37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5498, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_62d6740a9e054368a3ebab22d8f25119(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bf6b4653f28b9908d3a1b94fe0008c37
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d97ea42a0d9d1c52e45d437c1bbf306d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0048261284828186035, -0.07155326008796692, 0.013605579733848572, -0.051180481910705566], [0.14474661648273468, 0.2696493864059448, -0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, -0.3962211310863495, -0.15296784043312073], [0.14474661648273468, 0.2696493864059448, -0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, -0.3962211310863495, -0.15296784043312073], [0.22924979031085968, 0.1217564269900322, -0.07167693972587585, -0.07170835137367249], [0.22924979031085968, 0.1217564269900322, -0.07167693972587585, -0.07170835137367249]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_e1e861ac1ec4db0abc50373b09adf421(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1759, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_824b26dfd88deab0b0d563c534fe7ed8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e1e861ac1ec4db0abc50373b09adf421
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_44bcf06b903195105e76df11c140e77c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1538, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_43dc7cbfe2699a5a67b9bc91a2ea825a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44bcf06b903195105e76df11c140e77c
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_c8006008d0c9b50e9745f00920358122(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acaa3e3b1883649c3a05188c9dd575
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.07304361462593079, -0.10842274129390717, 0.23478034138679504, 0.023416876792907715], [0.03650416433811188, -0.2645467519760132, -0.3254188299179077, -0.08861064910888672], [0.26510629057884216, 0.05608522891998291, -0.21578019857406616, 0.10049128532409668], [0.07779538631439209, -0.06618337333202362, -0.022951990365982056, -0.24318337440490723], [0.07779538631439209, -0.06618337333202362, -0.022951990365982056, -0.24318337440490723], [0.26510629057884216, 0.05608522891998291, -0.21578019857406616, 0.10049128532409668]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_8940e98e7f5236af31bb2250fc33bdb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.013753771781921387, -0.2868351638317108, 0.03862294554710388, 0.19162918627262115], [-0.37577491998672485, -0.05430358648300171, 0.24721841514110565, -0.08289992064237595], [0.13008549809455872, -0.14730504155158997, 0.15147680044174194, -0.33067959547042847], [-0.2519821524620056, 0.363572895526886, -0.131232351064682, 0.2127522975206375], [0.013753771781921387, -0.2868351638317108, 0.03862294554710388, 0.19162918627262115]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_fa462f4f34f05640487a1ecae2a88f15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b23a2f4d39a9de5445545c7a96497782
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10226257890462875, 0.13534961640834808, -0.08023402094841003, 0.039990901947021484], [0.29901987314224243, -0.09267288446426392, -0.20913521945476532, 0.4559893012046814], [-0.2016167938709259, -0.13266520202159882, -0.10226882994174957, 0.20165568590164185], [0.008473038673400879, 0.02449806034564972, -0.06644713133573532, 0.26101258397102356]], dtype='float32').reshape([4, 4]),
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


    
    class PrimitiveOp_e3d0da78d5d4f8dad2657230e2d50bc7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2135, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d77465ffd0162f82e060228bde100dbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e3d0da78d5d4f8dad2657230e2d50bc7
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf9685eca2f0ecc1f6c24df0f02a40c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1575099527835846, -0.26694709062576294, -0.34720781445503235, 0.21742114424705505], [0.1575099527835846, -0.26694709062576294, -0.34720781445503235, 0.21742114424705505], [0.1509353071451187, -0.1447482705116272, -0.2221795916557312, 0.09455010294914246], [0.12404389679431915, -0.06935502588748932, -0.12357345223426819, -0.3634084165096283], [0.013999328017234802, -0.39129024744033813, -0.18125993013381958, 0.20690134167671204], [-0.006215885281562805, -0.10090631246566772, 0.3123472034931183, 0.14281310141086578], [-0.3736182451248169, 0.07743585109710693, 0.2678108513355255, 0.25866973400115967]], dtype='float32').reshape([7, 4]),
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


    
    class PrimitiveOp_4d7262f1590f938888dcb2f1492845f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4590, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_64e709934d3eb8b5c56dc9fa83086176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4d7262f1590f938888dcb2f1492845f2
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_70816090f789f2e9fa16f9b23e0ef620(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1042, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f819661a4fa3f443977ab40b93715d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70816090f789f2e9fa16f9b23e0ef620
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8433ed56ed5fd3b97cc3ed7df2f105d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_11acaa3e3b1883649c3a05188c9dd575
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1208452433347702, 0.1525149941444397, -0.18495747447013855, -0.21681168675422668], [0.016017157584428787, -0.0064589232206344604, -0.21185481548309326, -0.15630508959293365], [0.016017157584428787, -0.0064589232206344604, -0.21185481548309326, -0.15630508959293365], [0.35850846767425537, 0.1164940744638443, 0.17222779989242554, -0.30740097165107727], [0.02179768681526184, 0.04435417056083679, 0.2214246392250061, -0.19965055584907532], [0.01850026845932007, 0.061271607875823975, -0.1330414116382599, 0.09196221828460693]], dtype='float32').reshape([6, 4]),
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


    
    class PrimitiveOp_73b7e61510447fb9ea196e51b4e11b3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2339, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ae9e105b28946cf219e8875b6c3b5b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73b7e61510447fb9ea196e51b4e11b3e
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34a5c1d37e11bbe22d5965642a29e7d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3063, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a61e80e0ed1b0404098d1d2498fae54a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34a5c1d37e11bbe22d5965642a29e7d0
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_81d2c5a32a60f1f7314427bad136ef55(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[3822, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6698a1510e574cf4b216d2b892ebcd4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81d2c5a32a60f1f7314427bad136ef55
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
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


    
    class PrimitiveOp_d5f9c4f2f2bdfbc60605a1cf3a5ffb25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2057, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21bc4652cebf90f70d8e83fc42597c98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5f9c4f2f2bdfbc60605a1cf3a5ffb25
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_e329d6c89b8e071b1b3d270a28686344(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b03206884fb2651827882967eb6d09e6
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0021021366119384766, 0.10599559545516968, 0.1741461157798767, 0.11614590883255005], [-0.2877083122730255, 0.18331629037857056, -0.15134862065315247, 0.30544114112854004], [0.3105199337005615, -0.16365931928157806, 0.19212684035301208, -0.06973066926002502], [0.3105199337005615, -0.16365931928157806, 0.19212684035301208, -0.06973066926002502], [0.2566255033016205, 0.30786266922950745, -0.02622893452644348, -0.3976331949234009]], dtype='float32').reshape([5, 4]),
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


    
    class PrimitiveOp_3c6b2e5bc0b1489e7aafc2985aceae01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.abs(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4189, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6c25bd27a93f259b943b76ebc25583e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c6b2e5bc0b1489e7aafc2985aceae01
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c1d5486121a2494a5478e23381540bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba183f348bc3e9f5d7f8de36fe9deca3
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.026198573410511017, 0.122957244515419, 0.025363638997077942, -0.051180094480514526], [0.26046913862228394, 0.16602113842964172, 0.2520030736923218, -0.0859014093875885], [0.27233195304870605, -0.3711124062538147, -0.03736155107617378, 0.2736709713935852], [-0.026198573410511017, 0.122957244515419, 0.025363638997077942, -0.051180094480514526], [0.0075002312660217285, 0.25931796431541443, -0.013504356145858765, 0.38065314292907715], [-0.3905479311943054, -0.20227286219596863, 0.22368675470352173, 0.24777323007583618], [0.0075002312660217285, 0.25931796431541443, -0.013504356145858765, 0.38065314292907715]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_f3691c906c42ddcd9efef2ea5202c5ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1723, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94b9d1fdce4614421df4621de7c1a289(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.016139358282089233, 0.024570047855377197, 0.1798972338438034, 0.38466477394104004], [0.3155452609062195, -0.11460816860198975, 0.05521531403064728, -0.06962801516056061], [-0.045381829142570496, -0.20968413352966309, -0.11353392899036407, 0.1670221984386444], [0.3187718987464905, 0.2138129323720932, 0.054648011922836304, -0.12318618595600128], [0.14328435063362122, 0.12580473721027374, 0.005007922649383545, 0.018766164779663086]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_704c1dea06bbdffc1d60a06eac66a84f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.17177636921405792, -0.1657608598470688, 0.2102193832397461, -0.09437558054924011], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, -0.017763465642929077, -0.1887105405330658], [0.3667111396789551, 0.08710499107837677, 0.09478123486042023, 0.15412834286689758], [0.4154015779495239, 0.1597939133644104, -0.017763465642929077, -0.1887105405330658]], dtype='float32').reshape([5, 4]),
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


    class TestPrimitiveOp_236f5a999815e80a8c27f23f4d0316b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([5498, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51633851c21d969c82b18f276f0aa9f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.0048261284828186035, -0.07155326008796692, 0.013605579733848572, -0.051180481910705566], [0.14474661648273468, 0.2696493864059448, -0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, -0.3962211310863495, -0.15296784043312073], [0.14474661648273468, 0.2696493864059448, -0.01709042489528656, 0.3663172423839569], [0.29984790086746216, 0.004997730255126953, -0.3962211310863495, -0.15296784043312073], [0.22924979031085968, 0.1217564269900322, -0.07167693972587585, -0.07170835137367249], [0.22924979031085968, 0.1217564269900322, -0.07167693972587585, -0.07170835137367249]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_95ef9a10f41f638ed6d6cefaf3029351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1759, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_047ff9b88a8c7452c8b4ac57f3b15486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1538, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4809a8ae9ea3cb37c96edf5024fa6414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.07304361462593079, -0.10842274129390717, 0.23478034138679504, 0.023416876792907715], [0.03650416433811188, -0.2645467519760132, -0.3254188299179077, -0.08861064910888672], [0.26510629057884216, 0.05608522891998291, -0.21578019857406616, 0.10049128532409668], [0.07779538631439209, -0.06618337333202362, -0.022951990365982056, -0.24318337440490723], [0.07779538631439209, -0.06618337333202362, -0.022951990365982056, -0.24318337440490723], [0.26510629057884216, 0.05608522891998291, -0.21578019857406616, 0.10049128532409668]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_17c0383b6a6e7ccb0504d8f2a7c06be7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.013753771781921387, -0.2868351638317108, 0.03862294554710388, 0.19162918627262115], [-0.37577491998672485, -0.05430358648300171, 0.24721841514110565, -0.08289992064237595], [0.13008549809455872, -0.14730504155158997, 0.15147680044174194, -0.33067959547042847], [-0.2519821524620056, 0.363572895526886, -0.131232351064682, 0.2127522975206375], [0.013753771781921387, -0.2868351638317108, 0.03862294554710388, 0.19162918627262115]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_a071f6c6f1d23110a6910c304befcb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea1f431dcb50d4357a191377029e43c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.10226257890462875, 0.13534961640834808, -0.08023402094841003, 0.039990901947021484], [0.29901987314224243, -0.09267288446426392, -0.20913521945476532, 0.4559893012046814], [-0.2016167938709259, -0.13266520202159882, -0.10226882994174957, 0.20165568590164185], [0.008473038673400879, 0.02449806034564972, -0.06644713133573532, 0.26101258397102356]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_a836513d26728b27e45ec1c529b625b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([84, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_39332ad0de988a20b680d2a364ec1504(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2135, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7ea2d3ffcdc8dd5d5c9504c6e8eaeab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1575099527835846, -0.26694709062576294, -0.34720781445503235, 0.21742114424705505], [0.1575099527835846, -0.26694709062576294, -0.34720781445503235, 0.21742114424705505], [0.1509353071451187, -0.1447482705116272, -0.2221795916557312, 0.09455010294914246], [0.12404389679431915, -0.06935502588748932, -0.12357345223426819, -0.3634084165096283], [0.013999328017234802, -0.39129024744033813, -0.18125993013381958, 0.20690134167671204], [-0.006215885281562805, -0.10090631246566772, 0.3123472034931183, 0.14281310141086578], [-0.3736182451248169, 0.07743585109710693, 0.2678108513355255, 0.25866973400115967]], dtype='float32').reshape([7, 4]),
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


    class TestPrimitiveOp_be757d031fe538aea355a7d8b28936fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([4590, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_151371f6f354134d5e147a5093e82101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([1042, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72d39991db8df65da65e851dd8a198dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1208452433347702, 0.1525149941444397, -0.18495747447013855, -0.21681168675422668], [0.016017157584428787, -0.0064589232206344604, -0.21185481548309326, -0.15630508959293365], [0.016017157584428787, -0.0064589232206344604, -0.21185481548309326, -0.15630508959293365], [0.35850846767425537, 0.1164940744638443, 0.17222779989242554, -0.30740097165107727], [0.02179768681526184, 0.04435417056083679, 0.2214246392250061, -0.19965055584907532], [0.01850026845932007, 0.061271607875823975, -0.1330414116382599, 0.09196221828460693]], dtype='float32').reshape([6, 4]),
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


    class TestPrimitiveOp_53ce960f409d45577641197214e4aed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2339, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e9b89f0e2a152d2a15ce69b048066dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([3063, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8c00e9e094db94a166753f6ee8c90d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([3822, 4], dtype='float32', min=0, max=0.5),
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


    class TestPrimitiveOp_d6e570a0680d2a2522d8eca763458312(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([2057, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19f9833b15d9ffcc31b28175421ba45f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eaef0a1f4c8873a7876078fa4185f29a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f54a3285752cd8d40743bb3bed6deb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.0021021366119384766, 0.10599559545516968, 0.1741461157798767, 0.11614590883255005], [-0.2877083122730255, 0.18331629037857056, -0.15134862065315247, 0.30544114112854004], [0.3105199337005615, -0.16365931928157806, 0.19212684035301208, -0.06973066926002502], [0.3105199337005615, -0.16365931928157806, 0.19212684035301208, -0.06973066926002502], [0.2566255033016205, 0.30786266922950745, -0.02622893452644348, -0.3976331949234009]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_4fc8c070e10b8257efade3d932abe398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2559fb2d35a5b087ae373eadb3627aee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.uniform([4189, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d1a21df6e7fd9c9b79d90e1bcfb1b21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5eb1d5b3a7aff11948ee1d198d48f59c
        def get_inputs(self):
            return [
                paddle.to_tensor([[-0.026198573410511017, 0.122957244515419, 0.025363638997077942, -0.051180094480514526], [0.26046913862228394, 0.16602113842964172, 0.2520030736923218, -0.0859014093875885], [0.27233195304870605, -0.3711124062538147, -0.03736155107617378, 0.2736709713935852], [-0.026198573410511017, 0.122957244515419, 0.025363638997077942, -0.051180094480514526], [0.0075002312660217285, 0.25931796431541443, -0.013504356145858765, 0.38065314292907715], [-0.3905479311943054, -0.20227286219596863, 0.22368675470352173, 0.24777323007583618], [0.0075002312660217285, 0.25931796431541443, -0.013504356145858765, 0.38065314292907715]], dtype='float32').reshape([7, 4]),
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