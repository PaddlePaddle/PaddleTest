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
    class PrimitiveOp_098107b56224a3a7efa84540e7174b3f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2
            return paddle._C_ops.split_with_num(input_0, 2, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b203b4cd7e81646c433039336e6e73e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2
            return paddle._C_ops.split_with_num(input_0, 2, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_21086a656635e3d20e0f42be70cb8694(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_310c70ac417cd551c627206d31f37354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f44265f874c6f4a7fed012b5accc2ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2
            return paddle._C_ops.split_with_num(input_0, 4, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55ccb1743ffc4d82903c638673fcb63b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f44265f874c6f4a7fed012b5accc2ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 4, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e1193c1d1ed0e81d5a982427bddf121(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_679aec16085b9e30c5688c273344d60c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1
            return paddle._C_ops.split_with_num(input_0, 2, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_289d32a9596c2c2cfeff446270b0222c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1
            return paddle._C_ops.split_with_num(input_0, 4, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c050702e1fa37c4685526bdbc515ff0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c050702e1fa37c4685526bdbc515ff0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a941d19030a55d5245176e97bff1176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61e9bad7fd606f8679fdc1d1c605437e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_981fa7d2dffa26ddd0c5977ac225da36(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d6ecc1c3265c9c715c95dd5d9b88b4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21086a656635e3d20e0f42be70cb8694(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1
            return paddle._C_ops.split_with_num(input_0, 8, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa081ca71935af03a30a9d9e9ddf6e72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 224, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 1
            return paddle._C_ops.split_with_num(input_0, 4, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_26b4addb8566561902429aa2c6c3882b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([9, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26b4addb8566561902429aa2c6c3882b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([9, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f7f8e33771a6a42af81e9396a8e1944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e85066b7f478e18a7cddcc4503e1a966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b573efbb986d32287163cc30ac73122(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_608728feed5cc2d648d2620cea2dd38c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3a48f3efbad279519dff9411d9f60ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3a48f3efbad279519dff9411d9f60ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78591e71271c7a32748079ebeeaffd7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a09be532fda6993eda0688167bdccd5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 224, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5ecd0f18f0d0aa166d61567032307d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([10, 224, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_410579a14b44f5a940f45412ae0150cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 0
            return paddle._C_ops.split_with_num(input_0, 4, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_374fb0e4c4a8dc1a74e61859991021f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_410579a14b44f5a940f45412ae0150cf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936, 0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348, 0.45227816700935364, 0.2502661347389221, 0.16833464801311493, 0.15319602191448212, 0.18911296129226685, 0.1342129111289978, 0.27197179198265076, 0.4429328143596649, 0.17026685178279877, 0.08001313358545303, 0.23915867507457733, 0.05345135182142258], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_6153a7069cadbeda7dd4771d2212a978(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_410579a14b44f5a940f45412ae0150cf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.22442752122879028, 0.4289640784263611, 0.301647424697876, 0.0385211743414402, 0.3058600425720215, 0.14679817855358124, 0.3744381070137024, 0.0002531889476813376, 0.2294905185699463, 0.37590494751930237, 0.22168461978435516, 0.48501694202423096, 0.36503466963768005, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146, 0.3964419662952423, 0.30897533893585205, 0.40568625926971436, 0.10646888613700867, 0.4039801359176636, 0.40599721670150757], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_97723ebac824dcc27e088b1401266078(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97723ebac824dcc27e088b1401266078(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a941d19030a55d5245176e97bff1176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f14139ed73a4ea8889200add7ef6c6da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2
            return paddle._C_ops.split_with_num(input_0, 3, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_525b0ec03feb6f2f1d980392e945fdb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14139ed73a4ea8889200add7ef6c6da
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f7f8e33771a6a42af81e9396a8e1944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e85066b7f478e18a7cddcc4503e1a966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b573efbb986d32287163cc30ac73122(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2abe9b41cbc7744e15c27f0cce776c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af42219e8b22b47773f2c1eae3312e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8e7d2cb978925846078cd0f0dfa7dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaec30985d5175c0fde3ce834af77040(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 896, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_454f3e6c6f274e1c207407ee09b96928(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bdb6528a52b82b2549c5dee274c8924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bdb6528a52b82b2549c5dee274c8924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b070d98fc65e5a78848d39ebeca48b1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_708e355f0550fda618d12f158323ebb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 448, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21c9312db1895e3c8bf831cc836786df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([10, 112, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dab53eaf0a2e334eafc949afdaa04dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.178619846701622, 0.15484283864498138, 0.27537572383880615, 0.19937725365161896]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_7ce27c083495da55e106eb2cd707e5d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20313051342964172, 0.004780620336532593, 0.31011876463890076, 0.14600974321365356]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_b4d409b6f08188f75433ecda79eab508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3637081980705261, 0.2915404736995697, 0.28181594610214233, 0.4542856514453888], [0.39640137553215027, 0.06591049581766129, 0.256033331155777, 0.32371410727500916], [0.31833428144454956, 0.19010786712169647, 0.3946605324745178, 0.47006481885910034], [0.4412391483783722, 0.17498598992824554, 0.4794350862503052, 0.24301113188266754], [0.11331885308027267, 0.26003992557525635, 0.0947040542960167, 0.31819093227386475], [0.4164228141307831, 0.24949562549591064, 0.32221317291259766, 0.3337242603302002]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_540bfc3dadfc1e5834e01522375160db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.269661545753479, 0.25507786870002747, 0.0779203251004219, 0.4113561809062958], [0.24576835334300995, 0.1776416003704071, 0.28568974137306213, 0.38559842109680176], [0.09308407455682755, 0.0561361089348793, 0.2259289026260376, 0.34662380814552307], [0.3879702687263489, 0.35384827852249146, 0.08137910068035126, 0.25977855920791626], [0.43500569462776184, 0.1015116423368454, 0.37925606966018677, 0.09847179055213928], [0.14424768090248108, 0.23805175721645355, 0.45194125175476074, 0.15297983586788177]], dtype='float32').reshape([6, 4]),
            ]


    
    class PrimitiveOp_c7986ca8b9e9c6598786111827037529(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2
            return paddle._C_ops.split_with_num(input_0, 3, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_02f58d2edc86f35d2d4ae0fede6a1620(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7986ca8b9e9c6598786111827037529
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 1152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_47083903c875f7461d90000583a119da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2
            return paddle._C_ops.split_with_num(input_0, 3, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 576], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eedffc1b51aced9c8f55712802d5ad4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47083903c875f7461d90000583a119da
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a83d24eb96e1bb22d312060a995fd3db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a83d24eb96e1bb22d312060a995fd3db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d574d57442f2c842272e35c7a623665a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e850ef733a87ccec14091cc90d838b5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 112, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558106c738bd3ee25e32440d0dcfed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9558106c738bd3ee25e32440d0dcfed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_987374c6431dd80c90f3c1005389d4c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57b81e0fff7ffa2c32489758dc3cdfe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 34, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8504cf5032aaaf01c4d53152aad1c002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8504cf5032aaaf01c4d53152aad1c002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b0ddfffe6c36263a4be18038db5e2cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5ca496807d3b825cdc8cee7b136290a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adcf80585f79e4bdbaf1b275bf7b5cfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 448, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2af1ed84b26337c619d3cb145124816(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14139ed73a4ea8889200add7ef6c6da
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b94e45442e99d113281d94ff8ec021d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a66db429a3dd21ee44df93b68db0b8bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25348156690597534, 0.47471386194229126, 0.18775731325149536, 0.21230792999267578], [0.1419389396905899, 0.33389967679977417, 0.12440189719200134, 0.0743480697274208], [0.21522848308086395, 0.40402311086654663, 0.05388212949037552, 0.0162972342222929], [0.18140600621700287, 0.4485560953617096, 0.37235766649246216, 0.3133012354373932], [0.35213175415992737, 0.46584224700927734, 0.08130541443824768, 0.06640253961086273]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_bca52b43733f53dbde4bf08fa5eaa416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1649070680141449, 0.21730045974254608, 0.13528534770011902, 0.4135350286960602], [0.39336472749710083, 0.35416045784950256, 0.3804410994052887, 0.022249840199947357], [0.47650161385536194, 0.32481035590171814, 0.13431361317634583, 0.3079543709754944], [0.22819465398788452, 0.011503858491778374, 0.32307666540145874, 0.28616321086883545], [0.16766460239887238, 0.4248878061771393, 0.13232417404651642, 0.11313261836767197]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_39e7285efce63ce8030895b3abac967c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_843f449544de524bd927a8d660c65721(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2
            return paddle._C_ops.split_with_num(input_0, 3, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 288], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f20fef75c0462b989777f2bc79ec7f75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_843f449544de524bd927a8d660c65721
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4cf3abc269fd25ba91dd8b56769bb2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29d60e7bd02991b0fb0372f5c3220bf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 896, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42f86fb8e89b4e738e58ffb1e31adbae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42f86fb8e89b4e738e58ffb1e31adbae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47b1be56cc7c4b143746e68883082a4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78f37954598d5c4f96f45c6682228c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78f37954598d5c4f96f45c6682228c24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b94e45442e99d113281d94ff8ec021d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4647d0d9395132f6f5daf815243127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec4647d0d9395132f6f5daf815243127(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77d412455ea88ab33147e65d14495b8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a2f88eafeec487dfd22c9d0f71f252e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4259f4b0ed4cb312ab164173e1081664(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_843f449544de524bd927a8d660c65721
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38d96e0a254f5c333b9e33ebd48bd790(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47083903c875f7461d90000583a119da
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a50cb6fcc9e85b4532588ef519366f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.43446049094200134, 0.28952881693840027, 0.1857021450996399, 0.28646060824394226], [0.3114164471626282, 0.0999356284737587, 0.232812762260437, 0.2347613275051117], [0.21863189339637756, 0.3750646412372589, 0.44483304023742676, 0.07614689320325851], [0.22740739583969116, 0.1684504747390747, 0.43192264437675476, 0.21912746131420135]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_df07032033a7aad10f91e97b49072755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05875740945339203, 0.2514371871948242, 0.37367141246795654, 0.3861541450023651], [0.3892061412334442, 0.428699254989624, 0.26348745822906494, 0.4490680992603302], [0.20329052209854126, 0.0340176485478878, 0.22833964228630066, 0.4151049554347992], [0.3478715717792511, 0.16604164242744446, 0.3062015175819397, 0.03166484087705612]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_b5de5e7a2dc31dbab3a2cb897fd55a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7986ca8b9e9c6598786111827037529
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_511d797b93479901e96f44386daf05b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_511d797b93479901e96f44386daf05b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d574d57442f2c842272e35c7a623665a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eebc5176cd796b89fdd49e736293b1d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54ff99a6539130226dc9a57ad1c6352d
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca2ec56693fc7875c6a20ac3cdf96297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c06df30814cf6cca5a72f7278ae41b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c06df30814cf6cca5a72f7278ae41b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b87169e21209cf6c9efe53edd95f00b
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8e7d2cb978925846078cd0f0dfa7dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b203b4cd7e81646c433039336e6e73e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 21504, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a941d19030a55d5245176e97bff1176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_310c70ac417cd551c627206d31f37354(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55ccb1743ffc4d82903c638673fcb63b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f44265f874c6f4a7fed012b5accc2ae
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 4, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77d412455ea88ab33147e65d14495b8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47b1be56cc7c4b143746e68883082a4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_289d32a9596c2c2cfeff446270b0222c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3639ffb7a36b034eaa4d77526d5dcac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3639ffb7a36b034eaa4d77526d5dcac9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a941d19030a55d5245176e97bff1176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61e9bad7fd606f8679fdc1d1c605437e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8e7d2cb978925846078cd0f0dfa7dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d6ecc1c3265c9c715c95dd5d9b88b4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a941d19030a55d5245176e97bff1176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa081ca71935af03a30a9d9e9ddf6e72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 224, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26b4addb8566561902429aa2c6c3882b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([9, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26b4addb8566561902429aa2c6c3882b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([9, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f7f8e33771a6a42af81e9396a8e1944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e85066b7f478e18a7cddcc4503e1a966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b573efbb986d32287163cc30ac73122(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_608728feed5cc2d648d2620cea2dd38c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 320, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e2d1fffa0866f1b1a4e3430efe163e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e2d1fffa0866f1b1a4e3430efe163e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78591e71271c7a32748079ebeeaffd7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a09be532fda6993eda0688167bdccd5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 224, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5ecd0f18f0d0aa166d61567032307d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([10, 224, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_374fb0e4c4a8dc1a74e61859991021f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_410579a14b44f5a940f45412ae0150cf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.2341037541627884, 0.3048173189163208, 0.39980247616767883, 0.10136908292770386, 0.32081952691078186, 0.11360201984643936, 0.3236333429813385, 0.01610579900443554, 0.14992783963680267, 0.4712778925895691, 0.4077642858028412, 0.023976648226380348, 0.45227816700935364, 0.2502661347389221, 0.16833464801311493, 0.15319602191448212, 0.18911296129226685, 0.1342129111289978, 0.27197179198265076, 0.4429328143596649, 0.17026685178279877, 0.08001313358545303, 0.23915867507457733, 0.05345135182142258], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_6153a7069cadbeda7dd4771d2212a978(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_410579a14b44f5a940f45412ae0150cf
        def get_inputs(self):
            return [
                paddle.to_tensor([0.49831607937812805, 0.37963253259658813, 0.22442752122879028, 0.4289640784263611, 0.301647424697876, 0.0385211743414402, 0.3058600425720215, 0.14679817855358124, 0.3744381070137024, 0.0002531889476813376, 0.2294905185699463, 0.37590494751930237, 0.22168461978435516, 0.48501694202423096, 0.36503466963768005, 0.4213145971298218, 0.4012173116207123, 0.14370423555374146, 0.3964419662952423, 0.30897533893585205, 0.40568625926971436, 0.10646888613700867, 0.4039801359176636, 0.40599721670150757], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_3b7ce9902261f4023cbfc70705d405ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b7ce9902261f4023cbfc70705d405ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a941d19030a55d5245176e97bff1176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5781d22d14bffb60bdc399de72849c95(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = 2
            return paddle._C_ops.split_with_num(input_0, 3, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9710d2dce4412c92a0a6de3945be7df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5781d22d14bffb60bdc399de72849c95
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9f7f8e33771a6a42af81e9396a8e1944(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e85066b7f478e18a7cddcc4503e1a966(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b573efbb986d32287163cc30ac73122(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d574d57442f2c842272e35c7a623665a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af42219e8b22b47773f2c1eae3312e92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 34, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8e7d2cb978925846078cd0f0dfa7dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaec30985d5175c0fde3ce834af77040(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 896, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b94e45442e99d113281d94ff8ec021d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a213772a7b5b35b0cc0a242c36083287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a213772a7b5b35b0cc0a242c36083287(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b070d98fc65e5a78848d39ebeca48b1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_708e355f0550fda618d12f158323ebb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 448, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21c9312db1895e3c8bf831cc836786df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([10, 112, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dab53eaf0a2e334eafc949afdaa04dfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.178619846701622, 0.15484283864498138, 0.27537572383880615, 0.19937725365161896]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_7ce27c083495da55e106eb2cd707e5d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.20313051342964172, 0.004780620336532593, 0.31011876463890076, 0.14600974321365356]], dtype='float32').reshape([1, 4]),
            ]


    class TestPrimitiveOp_b4d409b6f08188f75433ecda79eab508(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.3637081980705261, 0.2915404736995697, 0.28181594610214233, 0.4542856514453888], [0.39640137553215027, 0.06591049581766129, 0.256033331155777, 0.32371410727500916], [0.31833428144454956, 0.19010786712169647, 0.3946605324745178, 0.47006481885910034], [0.4412391483783722, 0.17498598992824554, 0.4794350862503052, 0.24301113188266754], [0.11331885308027267, 0.26003992557525635, 0.0947040542960167, 0.31819093227386475], [0.4164228141307831, 0.24949562549591064, 0.32221317291259766, 0.3337242603302002]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_540bfc3dadfc1e5834e01522375160db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.269661545753479, 0.25507786870002747, 0.0779203251004219, 0.4113561809062958], [0.24576835334300995, 0.1776416003704071, 0.28568974137306213, 0.38559842109680176], [0.09308407455682755, 0.0561361089348793, 0.2259289026260376, 0.34662380814552307], [0.3879702687263489, 0.35384827852249146, 0.08137910068035126, 0.25977855920791626], [0.43500569462776184, 0.1015116423368454, 0.37925606966018677, 0.09847179055213928], [0.14424768090248108, 0.23805175721645355, 0.45194125175476074, 0.15297983586788177]], dtype='float32').reshape([6, 4]),
            ]


    class TestPrimitiveOp_82be962539bc99c1caca032a55de3ffe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5781d22d14bffb60bdc399de72849c95
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23606632683632558fc8929689341f41(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5781d22d14bffb60bdc399de72849c95
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d706d254e05a5c9b474800610f675368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d706d254e05a5c9b474800610f675368(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d574d57442f2c842272e35c7a623665a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e850ef733a87ccec14091cc90d838b5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 112, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dfb0b4a0ae99f5302eeea7cd28e54c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6dfb0b4a0ae99f5302eeea7cd28e54c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_987374c6431dd80c90f3c1005389d4c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57b81e0fff7ffa2c32489758dc3cdfe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 34, 160, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f76c1a1fadbc3bf17d5722068896ce6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f76c1a1fadbc3bf17d5722068896ce6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b0ddfffe6c36263a4be18038db5e2cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_987374c6431dd80c90f3c1005389d4c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_adcf80585f79e4bdbaf1b275bf7b5cfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 448, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5578e3d81ae0c79eecfbd4343385034d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5781d22d14bffb60bdc399de72849c95
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 2304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b94e45442e99d113281d94ff8ec021d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a66db429a3dd21ee44df93b68db0b8bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.25348156690597534, 0.47471386194229126, 0.18775731325149536, 0.21230792999267578], [0.1419389396905899, 0.33389967679977417, 0.12440189719200134, 0.0743480697274208], [0.21522848308086395, 0.40402311086654663, 0.05388212949037552, 0.0162972342222929], [0.18140600621700287, 0.4485560953617096, 0.37235766649246216, 0.3133012354373932], [0.35213175415992737, 0.46584224700927734, 0.08130541443824768, 0.06640253961086273]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_bca52b43733f53dbde4bf08fa5eaa416(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.1649070680141449, 0.21730045974254608, 0.13528534770011902, 0.4135350286960602], [0.39336472749710083, 0.35416045784950256, 0.3804410994052887, 0.022249840199947357], [0.47650161385536194, 0.32481035590171814, 0.13431361317634583, 0.3079543709754944], [0.22819465398788452, 0.011503858491778374, 0.32307666540145874, 0.28616321086883545], [0.16766460239887238, 0.4248878061771393, 0.13232417404651642, 0.11313261836767197]], dtype='float32').reshape([5, 4]),
            ]


    class TestPrimitiveOp_4b0ddfffe6c36263a4be18038db5e2cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db505952a00fca714bfd3aa3a9e15b5c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5781d22d14bffb60bdc399de72849c95
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4cf3abc269fd25ba91dd8b56769bb2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6b6c250123739b18d0b772c4e262aa5
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29d60e7bd02991b0fb0372f5c3220bf7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8a97ccf87357ee5e55c00c0394ef789
        def get_inputs(self):
            return [
                paddle.uniform([22, 896, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e7d2ca1ba5e1b9122c388718fe0ca38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e7d2ca1ba5e1b9122c388718fe0ca38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47b1be56cc7c4b143746e68883082a4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1fe9513c6076ef844ac51200a7c4a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1fe9513c6076ef844ac51200a7c4a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b94e45442e99d113281d94ff8ec021d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfcc29ff4074f70560df4eebe9fe71fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfcc29ff4074f70560df4eebe9fe71fc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_77d412455ea88ab33147e65d14495b8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78591e71271c7a32748079ebeeaffd7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_389f5efc062d7fdf22b5daf352587f5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5781d22d14bffb60bdc399de72849c95
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30a7d12d502e1ee80560edcb81ec4b19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5781d22d14bffb60bdc399de72849c95
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 576], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a50cb6fcc9e85b4532588ef519366f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.43446049094200134, 0.28952881693840027, 0.1857021450996399, 0.28646060824394226], [0.3114164471626282, 0.0999356284737587, 0.232812762260437, 0.2347613275051117], [0.21863189339637756, 0.3750646412372589, 0.44483304023742676, 0.07614689320325851], [0.22740739583969116, 0.1684504747390747, 0.43192264437675476, 0.21912746131420135]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_df07032033a7aad10f91e97b49072755(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.to_tensor([[0.05875740945339203, 0.2514371871948242, 0.37367141246795654, 0.3861541450023651], [0.3892061412334442, 0.428699254989624, 0.26348745822906494, 0.4490680992603302], [0.20329052209854126, 0.0340176485478878, 0.22833964228630066, 0.4151049554347992], [0.3478715717792511, 0.16604164242744446, 0.3062015175819397, 0.03166484087705612]], dtype='float32').reshape([4, 4]),
            ]


    class TestPrimitiveOp_efcf3943987c2a3df4930ef4657395dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5781d22d14bffb60bdc399de72849c95
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 1152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37b51cffb1c93037780d5d2c40c40bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37b51cffb1c93037780d5d2c40c40bbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d574d57442f2c842272e35c7a623665a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b070d98fc65e5a78848d39ebeca48b1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca2ec56693fc7875c6a20ac3cdf96297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86b30f62c7c79e2002dc38f3aa99ed33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86b30f62c7c79e2002dc38f3aa99ed33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d4f13a39ab3d4d721a73e116b0d7e57a
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8e7d2cb978925846078cd0f0dfa7dc1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_098107b56224a3a7efa84540e7174b3f
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()