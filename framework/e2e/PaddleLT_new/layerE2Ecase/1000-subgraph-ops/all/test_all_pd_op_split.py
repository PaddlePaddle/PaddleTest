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





if not (IsCinnStageEnableDiff() and LastCINNStageFailed()):
    class PrimitiveOp_afc3acd36208c066eef227cd33b6b453(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [18, 9]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 27, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0a26a7a9988c9abda393a1abc9e4ad4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86fff35b7b9ce5d39955426a20749454(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [4, 1]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6e1cab5df80651e8baf693e4e659836(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adf8f9c97320f7731798ff82894f0aca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4aad60815542199e991fdf98660b4dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [80, 32]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 112, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cecad1d34474be3d41037c4659fbe82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 6, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45d53d4c75cc238a052d0bf70da4cd46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_021ca12eb321220e6eac19c98a07a0b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a89a5dd52aee1fe7a1dc28f98fe6804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17bff82cfa07e11f8e5e3629184aa91c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c7f8f03171c8ec857ba05879c69006d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e24f4146e8f08b3ae13bdb7338eb2423(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [48, 48, 48]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_434a6c0ab85d7347ad62ee3d8e3bcf6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e24f4146e8f08b3ae13bdb7338eb2423
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c51c06ea9bf4f38f008780f9e0b7d2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2de51258b03d60bcf7a1a158978e9199(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b8bc8be9644a617934427cc6ef2775e1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 2, 1]
            input_2 = 2
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b660decf6bd4ca6e629f000e1e8fe1e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8bc8be9644a617934427cc6ef2775e1
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2746506929397583, 0.3692845404148102, 0.35885387659072876, 0.02352176047861576, 0.060231756418943405], [0.2529679834842682, 0.3538026511669159, 0.1643000692129135, 0.19557800889015198, 0.44981682300567627], [0.1918400079011917, 0.46419987082481384, 0.1283266544342041, 0.21114155650138855, 0.17508365213871002], [0.4446069300174713, 0.48151710629463196, 0.1822909265756607, 0.2833311855792999, 0.4978437125682831], [0.22252815961837769, 0.22237402200698853, 0.12502682209014893, 0.40927043557167053, 0.00514345383271575], [0.07176125049591064, 0.38622698187828064, 0.10549148172140121, 0.02151423506438732, 0.26088249683380127]]], dtype='float32').reshape([1, 6, 5]),
            ]


    class TestPrimitiveOp_0b997343a2aedf4cda90562c26f314d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72991126dbc43135b3f0090a99b70f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb3f008a8b1dfb97613b38182ae983ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7261a3b70a93371b5d369b7e92fb5626(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_665c24f14fdcfd9a42710c85a008a4e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [40, 40]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a454bfec9ddbdb21cb9a094f566c6c85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_665c24f14fdcfd9a42710c85a008a4e5
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d788e5b107195c4b246dbdd5bd053531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b997343a2aedf4cda90562c26f314d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4de6aa0b37e110ab2f7d71e265fbaf68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5682555e59b49a2e365dfcb8d3d664b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17bff82cfa07e11f8e5e3629184aa91c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e479ef710008abdd3ae416e3f07ca3ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1b021214905b2fa3445215c2eaab0bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_019d5fbdc1ffcf8a2fc61fba45290004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9f331a8d242dc526f5048841a530bc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 4, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2ffadd0b2a1c23648986ad5919eaefe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_019d5fbdc1ffcf8a2fc61fba45290004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de3c501e08f4cb92065da29c08fdd64c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_00a254df24285281eaa8e2b03c1f7d9f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [8, 8]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 16, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a12ea5288ef457e5bd0e7b69fbc7ad7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_00a254df24285281eaa8e2b03c1f7d9f
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_995847a7caba09aea3ed549e7dc6a3a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [48, 48]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b810370b35c83d1d6cf2841e6ae8a953(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_995847a7caba09aea3ed549e7dc6a3a1
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccc7b1dde490239989e3551c5bb9da79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f3d0ceb303e50401cea643ee946e4f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be63157121d565866a27db8b47c9a2c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [120, 120]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4ce747e654cfed291eb1c28390f6a4a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be63157121d565866a27db8b47c9a2c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06969034b180165a2d25afbd1de94f15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6a5bf183130e3035cfa2d195f1f4b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf0a25fb19a8214bd04d7e5c2e57a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a066734c9e46814214f575b6bd1ca474(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be63157121d565866a27db8b47c9a2c3
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_318c18bd4b339e9c2bd49a587f6f0675(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b2467246090ab120f22beee41df78be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17bff82cfa07e11f8e5e3629184aa91c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7261a3b70a93371b5d369b7e92fb5626(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a58ff80ba56165a1bf02fca268ecc77b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_791f82d803cca252708443a42bf29b28(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [16, 16, 32]
            input_2 = 3
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 49, 8, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f2e5d15d12aa555bc5b96111ad3b7e87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_791f82d803cca252708443a42bf29b28
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d26e8b953d95a496adbe0a77b87e0c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_778f0e318f8128daa01c092ac1cc2d02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [160, 160, 160]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7b0694f549ee0e3d569617f4188f422(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_778f0e318f8128daa01c092ac1cc2d02
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa40190ff80d9c2e9d3c8219ebf05691(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad8472fc994cccb39930c364150dc85f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42b412d97b601130ba2b81c088df5fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e479ef710008abdd3ae416e3f07ca3ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31f668b776feead8839823b3994eacf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4bbe833a3d31732b168903d56bec77a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [80, 80, 80]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a0b39400603fc39b215370fc36804ec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bbe833a3d31732b168903d56bec77a1
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42b412d97b601130ba2b81c088df5fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07e4c8e088bcdf7a168cb15bced636d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [16, 64]
            input_2 = 3
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 196, 8, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be66f315cca3047b51ee373c0acaf2b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07e4c8e088bcdf7a168cb15bced636d3
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 8, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b1f1ed86a1cd6b3482e4f81cd456894a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [300, 300, 300, 300]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1200, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5acb2b787c42f657237ce2474820c4d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1f1ed86a1cd6b3482e4f81cd456894a
        def get_inputs(self):
            return [
                paddle.uniform([22, 1200, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cecad1d34474be3d41037c4659fbe82d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 6, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5020b0058608e5063aefe7e0fe0c86e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42b412d97b601130ba2b81c088df5fb9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e64919501bf663fe2de360704639b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4ed2832cd14ac2116c58e19733e0dec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d7183e44af92e2821996a892dc99fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0a517a33c72b51ca9ff9feded90f7c8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [90, 90, 90, 90]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 360, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c8293e955f3efbfce50bab9f07d02fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0a517a33c72b51ca9ff9feded90f7c8
        def get_inputs(self):
            return [
                paddle.uniform([22, 360, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_612a8f459e3b55c8155d5273eab0c1bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_73b746d0cc24a155cb4336c2792d975e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [144, 144, 144, 144, 144]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 720, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b051cd9c0f5963c0997dd99b76ab7bfe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_73b746d0cc24a155cb4336c2792d975e
        def get_inputs(self):
            return [
                paddle.uniform([22, 720, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72991126dbc43135b3f0090a99b70f4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_911295e794d820aa7132a04f96e8bc19(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [600, 600]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1200, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1d5f79372e00cf2a2244da7e6a8eea9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_911295e794d820aa7132a04f96e8bc19
        def get_inputs(self):
            return [
                paddle.uniform([22, 1200, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1b021214905b2fa3445215c2eaab0bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7261a3b70a93371b5d369b7e92fb5626(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_291bbdcbab9fe4d432c831621caa24c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0877f41f2f84f4aa4de389612d69fa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a392d845a8ad39c738f29c6e41aeaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d8bf81e35eec48e5b81215cac19e230(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e38dd42dea32f5f1a7c578d673975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2a392d845a8ad39c738f29c6e41aeaa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f617905bf22ccd55d5135defdb84c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f2cf896f62f3eecf9bade48ac00344a3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [36, 36]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf2078d35703bf0e2c58eb1fdc8a8d87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f2cf896f62f3eecf9bade48ac00344a3
        def get_inputs(self):
            return [
                paddle.uniform([22, 72, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad8472fc994cccb39930c364150dc85f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bec68ec40273e126a9bdd3e1be9cff9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbc5a8e3dd74613ddcc9012258b465cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c86e631d8e1eda9e7b658113e65ad189(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_70f9327a897e12a03664827fb0c163e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [60, 60]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 120, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f16cffc2b9bc69b9d64d0b8734eecb9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_70f9327a897e12a03664827fb0c163e7
        def get_inputs(self):
            return [
                paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b514bd2f5da1279d26d31851d836ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6ca20fc1965c6bdf5fa5891639d2b3ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 1]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ecd92abefbb6a168e0695a9df0c159c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ca20fc1965c6bdf5fa5891639d2b3ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d03fe494f19da68dd2e53238ae7047ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [15200, 3800, 950, 247, 70]
            input_2 = 0
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20267, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6bfcee9d28073a4b0cd2c7ef6a482bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d03fe494f19da68dd2e53238ae7047ae
        def get_inputs(self):
            return [
                paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1c1e44c80087675b40230ee5981bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3807fab4fac45d8bb9f18749155ddc01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2ffadd0b2a1c23648986ad5919eaefe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4ba51a67ccbd56fe2cf09c65d473fefe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [20, 20]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4daa7a1c756609cfa2b1ac549068b0a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ba51a67ccbd56fe2cf09c65d473fefe
        def get_inputs(self):
            return [
                paddle.uniform([22, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a21e6ec57b7d31910a15b8a8ba6dde8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb3f008a8b1dfb97613b38182ae983ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ad8472fc994cccb39930c364150dc85f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bd714d21c21cb55c9e27ccb5a07f2cf8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [12, 12]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc2446479aa3da4335ce83f2e7729261(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bd714d21c21cb55c9e27ccb5a07f2cf8
        def get_inputs(self):
            return [
                paddle.uniform([22, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b8e6d7c5228d78526cdfe8287e710a5c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [180, 180]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 360, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd3656a92971d693b358c6b4ba7ae62d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8e6d7c5228d78526cdfe8287e710a5c
        def get_inputs(self):
            return [
                paddle.uniform([22, 360, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5242c6e306e14c4d6af8132d25b72a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2ffadd0b2a1c23648986ad5919eaefe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43b514bd2f5da1279d26d31851d836ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c51c06ea9bf4f38f008780f9e0b7d2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_92b5c92102ada302fabd77b70c70d01a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [16, 16, 32]
            input_2 = 3
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 196, 4, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4c6bd23f3f5534fc9ddc4a2e81d54257(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92b5c92102ada302fabd77b70c70d01a
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c5a82a5a76a881cac93dd74259b8ed38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [240, 240]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d06a46ab564d20f7d1d48c2c2ec6204b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5a82a5a76a881cac93dd74259b8ed38
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a1a174dd47f13c4999b343f40ffb3ddd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [16, 16, 32]
            input_2 = 3
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 16, 12, 64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_25abe49a6ac0efc7f8dbb71223cd1753(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1a174dd47f13c4999b343f40ffb3ddd
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48684c1af9b296e07a0d50786c637477(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4aad60815542199e991fdf98660b4dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1cfde6b0e6ae38fb47ac93d9241e2e39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_afc3acd36208c066eef227cd33b6b453
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2826b1d4c845ea729c98ff7c7ae2e565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e4d29de52c91f0543b8590d1851fb77a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ca20fc1965c6bdf5fa5891639d2b3ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f70bc7cf1127ed1e01d3bde818b90209(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [16, 64]
            input_2 = 3
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[22, 49, 16, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e8e7db940bf8c7107bdfd5a671a224c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f70bc7cf1127ed1e01d3bde818b90209
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 16, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_08a12289273734dfec7959e77b1559c6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [18, 9]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebe7b688a99f2b125e9293c040b08ed5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f6e1cab5df80651e8baf693e4e659836(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4a68a5eade7aafbeda14e291cd4de77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_22633262e36a0854cc9cf7832d69704f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [80, 32]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32d4fd892d3bc8012a5fedb44ab41cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 6, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fab0dda6a6f0b317252d1ba90ec317e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_021ca12eb321220e6eac19c98a07a0b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bddabd051e7ec47873cf92ec2bfa3e2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e16351fe3e0e0847eb558408a3c4084(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5649a58e72a4f991251f40ef136ecc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 11, 11], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7afc090ac1c4eb856ab879e96b64f849(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [48, 48, 48]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_60d408de0d69211d3b3c96c7e9ab6c20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7afc090ac1c4eb856ab879e96b64f849
        def get_inputs(self):
            return [
                paddle.uniform([22, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68698f34bc988ea9f54de98d2d672521(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3d78da0787280d30fab7b48c251a6b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 9, 9], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b660decf6bd4ca6e629f000e1e8fe1e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8bc8be9644a617934427cc6ef2775e1
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2746506929397583, 0.3692845404148102, 0.35885387659072876, 0.02352176047861576, 0.060231756418943405], [0.2529679834842682, 0.3538026511669159, 0.1643000692129135, 0.19557800889015198, 0.44981682300567627], [0.1918400079011917, 0.46419987082481384, 0.1283266544342041, 0.21114155650138855, 0.17508365213871002], [0.4446069300174713, 0.48151710629463196, 0.1822909265756607, 0.2833311855792999, 0.4978437125682831], [0.22252815961837769, 0.22237402200698853, 0.12502682209014893, 0.40927043557167053, 0.00514345383271575], [0.07176125049591064, 0.38622698187828064, 0.10549148172140121, 0.02151423506438732, 0.26088249683380127]]], dtype='float32').reshape([1, 6, 5]),
            ]


    class TestPrimitiveOp_4c70b3b2c25df1722e9d916399f3c005(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63fe88bc1f890053a6987f3696a67ef8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00c4c9fb38d792bc6383c5117edf7b1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ae14c442847fd3dcf47ebaaa099e94f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_df49ca3e8bdd744ef7b2d0fa9323f9b8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [40, 40]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_049dad30c676e7acc9ae48c5cfcd39cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df49ca3e8bdd744ef7b2d0fa9323f9b8
        def get_inputs(self):
            return [
                paddle.uniform([22, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d788e5b107195c4b246dbdd5bd053531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c70b3b2c25df1722e9d916399f3c005(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c30ec8a16f827bca1a7d1680e011af0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5682555e59b49a2e365dfcb8d3d664b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e16351fe3e0e0847eb558408a3c4084(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a3b210fbfc74fa632ae1928ac9fa5b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce9af32387f2ad8cba0e74178eca3812(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_092443a9d0064ea4040a17e4f863ec9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45f2396b33c5ced7897c7e44baf236c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 4, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e052e38363e232f410af63167f053131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_092443a9d0064ea4040a17e4f863ec9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bb41782af6e9de460cf0261f1019077(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ef54b8f3f67547107b8a8b14cfc13281(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [8, 8]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_136541eda6dd5afbe04a67b7a70a6db3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ef54b8f3f67547107b8a8b14cfc13281
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c0e1650b35dc08d3df24d250fa9ec4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [48, 48]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45d0278aef6d10788aeef4d365fbf8ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c0e1650b35dc08d3df24d250fa9ec4c
        def get_inputs(self):
            return [
                paddle.uniform([22, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccc7b1dde490239989e3551c5bb9da79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86096a4b1ad2cf84d664e69d414712c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_06f3885ea9e9ed9d159797628aa91690(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [120, 120]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96f878e97301cfe6fe932e0c145b43b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06f3885ea9e9ed9d159797628aa91690
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c32f83c2a7ecebe41b484f5cd6b37e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6a5bf183130e3035cfa2d195f1f4b74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf0a25fb19a8214bd04d7e5c2e57a1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_813bc071f8fed58c6345a89eb1c9e131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06f3885ea9e9ed9d159797628aa91690
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bec0ded52401d2bde5716d893346ab11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_744d115e9064c8d86f86867421d47541(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 5, 5], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e16351fe3e0e0847eb558408a3c4084(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ae14c442847fd3dcf47ebaaa099e94f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab05cd7eec9221cec5d1792b682cb536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0ccbf4284280ab3ac966e20421a730bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [16, 16, 32]
            input_2 = 3
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_48e6fd6647d70ecebdf83385ce99fc2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ccbf4284280ab3ac966e20421a730bd
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 8, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ddc392b2dd386961cbe2816c555f04c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d02300bf98890e7c9d5345f1b6e39eef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [160, 160, 160]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e051985be3dd96e2bfd174d060b1470f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d02300bf98890e7c9d5345f1b6e39eef
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe8793a10c6316009ee8ae10ef3dd42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e18976f9fd165dd5201688190a8e1250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dff4ebc105874c8d335c387388d4a517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a3b210fbfc74fa632ae1928ac9fa5b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb8d47b23e8e7ae6c0c3ab8913c11963(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb7998e5f98e808ae555b8b88677c851(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [80, 80, 80]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3088704bbd20caa29105adb9d9c47399(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb7998e5f98e808ae555b8b88677c851
        def get_inputs(self):
            return [
                paddle.uniform([22, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dff4ebc105874c8d335c387388d4a517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_71a93175eaf5f11197ba436a202b176f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [16, 64]
            input_2 = 3
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03d2cb656150ed2e32d1a17f4f148839(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71a93175eaf5f11197ba436a202b176f
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 8, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b60a52a9b0b8ba836e9bc1da13eb04ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [300, 300, 300, 300]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d57638cb97e59ff6b2e1eb1bc3fd5032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b60a52a9b0b8ba836e9bc1da13eb04ae
        def get_inputs(self):
            return [
                paddle.uniform([22, 1200, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32d4fd892d3bc8012a5fedb44ab41cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 6, 6], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c9c8ad6d4e3d437157ba97da098f437(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dff4ebc105874c8d335c387388d4a517(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17e64919501bf663fe2de360704639b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_82b1eacae2f759b320adc1b92a2bb716(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d7183e44af92e2821996a892dc99fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bdb35110570014740e957fe92e7bc26c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [90, 90, 90, 90]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98f23b345aae81cb215340e249492083(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bdb35110570014740e957fe92e7bc26c
        def get_inputs(self):
            return [
                paddle.uniform([22, 360, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_612a8f459e3b55c8155d5273eab0c1bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5bfb8bf404bb183a094c12ef3b977468(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [144, 144, 144, 144, 144]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c0fc2fc39a988a0b74bef401f7bb815(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5bfb8bf404bb183a094c12ef3b977468
        def get_inputs(self):
            return [
                paddle.uniform([22, 720, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63fe88bc1f890053a6987f3696a67ef8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_31b99ff92373100e37107420962e1f32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [600, 600]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83bfb0479e0f5449a92486f146d76f73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31b99ff92373100e37107420962e1f32
        def get_inputs(self):
            return [
                paddle.uniform([22, 1200, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce9af32387f2ad8cba0e74178eca3812(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ae14c442847fd3dcf47ebaaa099e94f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_291bbdcbab9fe4d432c831621caa24c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a6da72da1142900f5c8123166efbc85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef0dcf7264ddf062df566d13b910f8ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05fb6e8c0e51566846f9d7eaacd36f04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e38dd42dea32f5f1a7c578d673975fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef0dcf7264ddf062df566d13b910f8ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62f617905bf22ccd55d5135defdb84c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b619efc628e7a660a13626a76d5e16e4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [36, 36]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_52dd1ccf17071a521e9d2f545a3e0eba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b619efc628e7a660a13626a76d5e16e4
        def get_inputs(self):
            return [
                paddle.uniform([22, 72, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e18976f9fd165dd5201688190a8e1250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ac1f0fb4d28c384254cc5c71283cbd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f36290e057bcd8d902cc85fd93dbc5fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45404a4e152c107fe56046fac44c21b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe4e4d22b02e014f5618bbb3862e11c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [60, 60]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_700a33db9b61b431959d2b9fcddf706b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe4e4d22b02e014f5618bbb3862e11c1
        def get_inputs(self):
            return [
                paddle.uniform([22, 120, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d2d9bac607df60e9988e4d3a66b7e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b26ebf77b1df39c34b5eced6a0ad9dd3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [2, 1]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0dd9f74fc97a82e50565d26a338a7a36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b26ebf77b1df39c34b5eced6a0ad9dd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_06ada3b837e2497cee95511621561d9d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [15200, 3800, 950, 247, 70]
            input_2 = 0
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9eae0ab96e429e92b2d342e7e0d200cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06ada3b837e2497cee95511621561d9d
        def get_inputs(self):
            return [
                paddle.uniform([20267, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1c1e44c80087675b40230ee5981bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8752c60be5d34fc408cca1de107d6413(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e052e38363e232f410af63167f053131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf4ab5883652501a8e19d819515a9496(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [20, 20]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41db9c1c25a4f64f2323e6e071e77ffb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf4ab5883652501a8e19d819515a9496
        def get_inputs(self):
            return [
                paddle.uniform([22, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7edabe2719cc1cfaa6e2b02ff069ba7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00c4c9fb38d792bc6383c5117edf7b1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e18976f9fd165dd5201688190a8e1250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_252006fd12af4572d70a3702dcf453de(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [12, 12]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02ba1dd83567b91fc83bb9182e6c9bc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_252006fd12af4572d70a3702dcf453de
        def get_inputs(self):
            return [
                paddle.uniform([22, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d07089b67574fc10528a2b8f0bdaa98(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [180, 180]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_42a962742d9114563dc1ad1cbca66577(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d07089b67574fc10528a2b8f0bdaa98
        def get_inputs(self):
            return [
                paddle.uniform([22, 360, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63d215bd8b4af94fa1d8cfb259b7c12d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e052e38363e232f410af63167f053131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34d2d9bac607df60e9988e4d3a66b7e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68698f34bc988ea9f54de98d2d672521(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4f3f8879a592c86a3861021abf94bb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ccbf4284280ab3ac966e20421a730bd
        def get_inputs(self):
            return [
                paddle.uniform([22, 196, 4, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7c8b68f63bab72d295a10ebb1db1e6cc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [240, 240]
            input_2 = 1
            return paddle.split(input_0, input_1, input_2)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_33e9814a14e2f7d8acbd6a3edc26dc36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c8b68f63bab72d295a10ebb1db1e6cc
        def get_inputs(self):
            return [
                paddle.uniform([22, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71ac9bfa98ed422b31cf206c8665a2e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0ccbf4284280ab3ac966e20421a730bd
        def get_inputs(self):
            return [
                paddle.uniform([22, 16, 12, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_838d5c6e30d2d286248b1f4d4b475388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22633262e36a0854cc9cf7832d69704f
        def get_inputs(self):
            return [
                paddle.uniform([1, 112, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c44888c99323706b34cfe6224d468aba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a12289273734dfec7959e77b1559c6
        def get_inputs(self):
            return [
                paddle.uniform([1, 27, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2826b1d4c845ea729c98ff7c7ae2e565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86fff35b7b9ce5d39955426a20749454
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84acabbf75c41a711467daa034c04f83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b26ebf77b1df39c34b5eced6a0ad9dd3
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b77cb1f00245a0850f2aa7c5eeee178(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_71a93175eaf5f11197ba436a202b176f
        def get_inputs(self):
            return [
                paddle.uniform([22, 49, 16, 80], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()