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
    class PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd4b2cae00166cb134348f1795b763e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 9, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06ebb6082105656294c021516e1ae42e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bbbee578c568a31cd82396e1b35a1a18(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e300584622ca777af376081f450842c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbee578c568a31cd82396e1b35a1a18
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ef96babd87221c2dd52db48c2be1138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3092188f29a6dcbc421f39e7e756050c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1e89d13e273b9f8ebfcf9f0bc57b3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d721281f7f446b2372c521e2e2a81ef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58ffa96b5cac1b346d18e38a854299b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_20caea08a4b2918efea7648f5d8fffe9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 960, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0d082c503e35ebc0492ccdaa85f3155(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_20caea08a4b2918efea7648f5d8fffe9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd880d1e9e39721be50e486875773de8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acef13723a8d979e3a31c8ff12a462ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ce4350a5c695e334da8fe49aba28a03a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae7eedf4fb30b5986a15c1ac56dd8293(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce4350a5c695e334da8fe49aba28a03a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.212127923965454]], [[1.789625644683838]], [[1.933072566986084]], [[2.306485652923584]], [[1.1387118101119995]], [[2.362990140914917]], [[2.7894678115844727]], [[2.110621690750122]], [[1.9468934535980225]], [[1.6261770725250244]], [[2.458838939666748]], [[1.5788171291351318]], [[2.1331140995025635]], [[1.3116607666015625]], [[2.6825926303863525]], [[1.6499090194702148]], [[3.0373640060424805]], [[2.0838427543640137]], [[1.774763822555542]], [[1.7819828987121582]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_39141bf35ea856e49e5e26de7cf95ff9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8e81e5a381fa379dccfbe4ea47049e67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39141bf35ea856e49e5e26de7cf95ff9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a2b5bb12a0f08219c8721b9b9ddb4ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2f06854817eecaceaf98de639f1ffe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c94ab3200a3bd5a6ee8671630b4bcb5e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de542f49cd8e8c08c60272d452e3dba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c94ab3200a3bd5a6ee8671630b4bcb5e
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2e02d184efd8c44aff21e51047c8909e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3fbf5a40bc3f5a00bd6c1da351477a95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24471a510fc64e2e28874eb33e01ae56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e300584622ca777af376081f450842c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbee578c568a31cd82396e1b35a1a18
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17ddb8421d93af333845334dfd0b203d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de542f49cd8e8c08c60272d452e3dba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c94ab3200a3bd5a6ee8671630b4bcb5e
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe8d67bef90555fc81e8965cb8299b22(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1152, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_caee6dddffc9796c44291a81d9b12383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d67bef90555fc81e8965cb8299b22
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8fdc543baed6e98fab30fb8f424b65ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb92727a25d743927f7520ebfb6b8d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1192f0c96a780df9d800f10109adb625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2e609b8a0e3f37646c61d3fafc04d1d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84ae02993732a79fa0a6b9a769fae398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e609b8a0e3f37646c61d3fafc04d1d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c061331fd2b874321a2fb7091c31e04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78b3162838f7f8e0572f39cdb8dc04bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e1f1b81f0b991ad077af7d7e86045d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_65e5b9402fd2e9062d9cad1fb07d731c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 15, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6684fdcf7bc43b505242d57d3d4d9eed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65e5b9402fd2e9062d9cad1fb07d731c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7841f353ce7332e4f6f21b4bbe4fd86b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be0cd4f8c2d1d0b301daf54e7b3f995c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45a265cf30a0d0e1dbcb77dedf0e546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3f75e2631425f035f9f68c63d175d44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b02870473dc7de7fd2f891669923976a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c283fa118513e6ebd8b20cf1b8823103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b02870473dc7de7fd2f891669923976a
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f795edd1d030a261be7723256147673c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d721281f7f446b2372c521e2e2a81ef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c9870beeec61c0efddc845d9c913cce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e300584622ca777af376081f450842c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbee578c568a31cd82396e1b35a1a18
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9463c504151e77cba49829843806258(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c3a08495da1ef8bb9f3d8e1781fc22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d5a79e5948586ca5db8d6beef7a3131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65e5b9402fd2e9062d9cad1fb07d731c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17ddb8421d93af333845334dfd0b203d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1cf7698bae56cd95fb96a1b2006c154(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65e5b9402fd2e9062d9cad1fb07d731c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be0cd4f8c2d1d0b301daf54e7b3f995c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65750189b9ee00391ddd8f88320547fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0a75a51bac1ec7d6df6d1d943edfe2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_805deca12a4e89db1d64eea7e0a5248f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_018c28c6db1828c1248151c4caca5be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_805deca12a4e89db1d64eea7e0a5248f
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cabeb3be71d0f242dc47615c3f0cefd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c310041ad5b906560b8bf7984b812053(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f9a5123c62e548fb9069aa7361093f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c310041ad5b906560b8bf7984b812053
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c8bcabcd402a65525736ae588bfb1a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275ad47bdac9febafabbb5980cbd6ad4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cabeb3be71d0f242dc47615c3f0cefd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d0388a06c13787855a5f5f5e602e699(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58ffa96b5cac1b346d18e38a854299b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaa7d37e281eeca2ea888f2e24a25b00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3f75e2631425f035f9f68c63d175d44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75c15f2c13a0dd5a09f9185cbdfc0cf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61d5132c96c98615f6d778896cc30882(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5cf328e9ef2770850446da0d88583a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb7027f600b3b2248fecca41ed8c7848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c95694eb664c20a5c9b18994403cd35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fdf7d495bee27942e2dc21dc722a6732(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f4d51d1f7d3329b671a92b771c5d8e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdf7d495bee27942e2dc21dc722a6732
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f84f50a9c2fedc8344c1d2dd758e94b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4640ad20b6e7628608ea2669b318ccb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8764ec62403792b37f34ad20d575cd46(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_51d791455990b6f66fb97903fe63a467(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de542f49cd8e8c08c60272d452e3dba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c94ab3200a3bd5a6ee8671630b4bcb5e
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31463778049a5ebd552fbceb6fb6708e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24471a510fc64e2e28874eb33e01ae56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70702e66f9249b269e550c165e9d7c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a589eedcb25a0d94cecb4a372edebba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdf7d495bee27942e2dc21dc722a6732
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a55c74ebb313a43a6eb5a4cde5ac0bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbee578c568a31cd82396e1b35a1a18
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f9a5123c62e548fb9069aa7361093f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c310041ad5b906560b8bf7984b812053
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17ddb8421d93af333845334dfd0b203d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7841f353ce7332e4f6f21b4bbe4fd86b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfa3d985769f2aa448fbe38ef0b86161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d67bef90555fc81e8965cb8299b22
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d9acb30993fa3d7f568459a5ffe80ea2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eec1e6278095e36d2bd419cdae772ddd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9acb30993fa3d7f568459a5ffe80ea2
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc11b487997c6fa970cd426ecaddf35b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f84f50a9c2fedc8344c1d2dd758e94b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0a23c0e7dc7a3a3306ecfd8daafcf87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c310041ad5b906560b8bf7984b812053
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6cb40d3794db55561c0167f9b362a566(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecc57ed456976e3f8f5e6152ca62f22f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_40ff7cae2711788f20bfa5bb80cc8860(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 13, 19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_789222df6e38dcfcf693db1250ceef91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_40ff7cae2711788f20bfa5bb80cc8860
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03027e005a4a9aac15a14c0fe8ab2fb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16158dcd321a5682937153b31996ee8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd4b2cae00166cb134348f1795b763e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a55c74ebb313a43a6eb5a4cde5ac0bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbee578c568a31cd82396e1b35a1a18
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecc57ed456976e3f8f5e6152ca62f22f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a589eedcb25a0d94cecb4a372edebba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdf7d495bee27942e2dc21dc722a6732
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd4b2cae00166cb134348f1795b763e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b7844cfee339aabd3d6f2cc94fb0325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc34da914c49024c46776b0e71772026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b257955bf4a6063f4e97d0790dc3c88a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c94ab3200a3bd5a6ee8671630b4bcb5e
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ecc57ed456976e3f8f5e6152ca62f22f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c5ddc74894f155651585d12c8e71597(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5396bfcd914b183ed29797404f56769(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_434eb366a10cc8c26042c6863d0ee938(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb1032b90122dfca9f8ed1212b1e2f15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e06a9a9b2f05ad6fe4fc38dc9eb80b7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53985539acb423cf71b56af59f3ff2fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e1f1b81f0b991ad077af7d7e86045d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01b2df4c8acb1765e32dccf862a00b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9acb30993fa3d7f568459a5ffe80ea2
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4d51d1f7d3329b671a92b771c5d8e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdf7d495bee27942e2dc21dc722a6732
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5ab4984e5bd7201655ec6ce4c5c88dd8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_96108c82ba7d123fb4b354994069e3ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ab4984e5bd7201655ec6ce4c5c88dd8
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91db9c5ea8476a1f2769aae5265e052b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65e5b9402fd2e9062d9cad1fb07d731c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc34da914c49024c46776b0e71772026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e514c338ecd9c49d7a401b54d312298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be0cd4f8c2d1d0b301daf54e7b3f995c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb92727a25d743927f7520ebfb6b8d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b349fef70e301c380190099f2e33537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24471a510fc64e2e28874eb33e01ae56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45a265cf30a0d0e1dbcb77dedf0e546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45a265cf30a0d0e1dbcb77dedf0e546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16ea2a8e30d2324ff09926cb0de36ed4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275ad47bdac9febafabbb5980cbd6ad4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cc8a25731ea6fdf0aacb9adf6c0fa94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebc725b4ec959ef4c24d6b67644ffccc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd4b2cae00166cb134348f1795b763e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4d51d1f7d3329b671a92b771c5d8e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdf7d495bee27942e2dc21dc722a6732
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0a75a51bac1ec7d6df6d1d943edfe2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84ae02993732a79fa0a6b9a769fae398(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e609b8a0e3f37646c61d3fafc04d1d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfa3d985769f2aa448fbe38ef0b86161(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d67bef90555fc81e8965cb8299b22
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7841f353ce7332e4f6f21b4bbe4fd86b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a589eedcb25a0d94cecb4a372edebba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdf7d495bee27942e2dc21dc722a6732
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfa0f168c53b0853f9e4fc7621cbedb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16cc225448d83d7fe904f377918cb380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_671cde9fb259cbd7baefe3bae31796ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5da13faa8c8267511fca3cc01094a749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ceb710bc3e4e850636ee950ccc0a461c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb92727a25d743927f7520ebfb6b8d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8a0eb3b641439785badb91a22dccd52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f5e97c61a630f7448ccf4dda6b92589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5da13faa8c8267511fca3cc01094a749(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e300584622ca777af376081f450842c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbee578c568a31cd82396e1b35a1a18
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de542f49cd8e8c08c60272d452e3dba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c94ab3200a3bd5a6ee8671630b4bcb5e
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4cfbef429f9703eb9a85d65409b02e88(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ded88b6c76d45608ff82053638738ebc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cfbef429f9703eb9a85d65409b02e88
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4571b63363b1db383087d0774f800fd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2a3cdd57b2386031e1895fe4732cf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a3fe9ab6e6e04cbc954a735897f9de3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_717f62e10669f8ffa05621258e92eb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ded88b6c76d45608ff82053638738ebc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4cfbef429f9703eb9a85d65409b02e88
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a63e0213c6cfd45c999bbe1a22fb8f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6cb40d3794db55561c0167f9b362a566(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b257955bf4a6063f4e97d0790dc3c88a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c94ab3200a3bd5a6ee8671630b4bcb5e
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d721281f7f446b2372c521e2e2a81ef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c0658f45fa02e00aa443f3c2cee4ea3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c6a54f14c4f427a0c143b1258575004(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c0658f45fa02e00aa443f3c2cee4ea3
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.7184655666351318]], [[1.7496981620788574]], [[2.094937324523926]], [[2.0464823246002197]], [[2.109078884124756]], [[1.7068365812301636]], [[1.2507412433624268]], [[2.1731138229370117]], [[1.676121711730957]], [[1.1951981782913208]], [[2.1150741577148438]], [[1.6924262046813965]], [[1.820841908454895]], [[0.9820704460144043]], [[2.175051212310791]], [[2.179396867752075]], [[1.4661563634872437]], [[1.5999906063079834]], [[1.7346348762512207]], [[2.3203139305114746]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    
    class PrimitiveOp_dd6daa4e0cb6c37def9897e409e6c483(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 40, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82a01da8a6b1df03e5dab0ba8fa9594d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd6daa4e0cb6c37def9897e409e6c483
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_18685c8bdf2233fcf62239ddf6025459(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a5d24551fb7a75727a868b7ed9c81375(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18685c8bdf2233fcf62239ddf6025459
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d47129ebe99bc1941a7934c57860cf5f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 160, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9508332525e089b5f8a79288aa1d017(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d47129ebe99bc1941a7934c57860cf5f
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ec7cc57ee89157dd3dffa811e1a0423(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c105951c1eeb1d3f4405bd5c892131e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_805deca12a4e89db1d64eea7e0a5248f
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_530d878a3643a655a912672d7a804295(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a55c74ebb313a43a6eb5a4cde5ac0bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbee578c568a31cd82396e1b35a1a18
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e1f1b81f0b991ad077af7d7e86045d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aad6b8b6c804ab267f73cf01130188b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65e5b9402fd2e9062d9cad1fb07d731c
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7a70374a64ddc146454769d7eec7a2dc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 50, 76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc0b5e1b091ae50da199f1d704f967a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7a70374a64ddc146454769d7eec7a2dc
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ae0b456a4bab379079bccd314ad96928(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b6f64f7a4ee3b4c63300bfaec10f4142(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae0b456a4bab379079bccd314ad96928
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65750189b9ee00391ddd8f88320547fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_055471b20041bee156c7542aae30a9da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6d2fa10fca2c68a739dc5577027c4de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d567f31442c58e3d01dd49f95cce099c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e867838d3a6db0058cd1df608b63e5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d567f31442c58e3d01dd49f95cce099c
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c8bcabcd402a65525736ae588bfb1a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45a265cf30a0d0e1dbcb77dedf0e546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b29c1fc8a68b4354b6d596f590251fed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ce4350a5c695e334da8fe49aba28a03a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.7839205265045166]], [[2.3346548080444336]], [[2.28593373298645]], [[2.7708616256713867]], [[2.30279278755188]], [[3.2229268550872803]], [[2.696528911590576]], [[2.943495988845825]], [[3.1310055255889893]], [[1.8862913846969604]], [[2.6206557750701904]], [[2.277665853500366]], [[1.9456156492233276]], [[1.5896580219268799]], [[2.2188262939453125]], [[2.2632229328155518]], [[1.6459977626800537]], [[3.8634533882141113]], [[1.8549753427505493]], [[3.404752492904663]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_8e81e5a381fa379dccfbe4ea47049e67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39141bf35ea856e49e5e26de7cf95ff9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0fb814414d3edcb57c406b9dece44f01(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6b2c099e52e303db130a511263c1a7be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0fb814414d3edcb57c406b9dece44f01
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c105951c1eeb1d3f4405bd5c892131e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_805deca12a4e89db1d64eea7e0a5248f
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fbf5a40bc3f5a00bd6c1da351477a95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a55c74ebb313a43a6eb5a4cde5ac0bd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bbbee578c568a31cd82396e1b35a1a18
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4d51d1f7d3329b671a92b771c5d8e25(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdf7d495bee27942e2dc21dc722a6732
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6cb40d3794db55561c0167f9b362a566(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24471a510fc64e2e28874eb33e01ae56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b71e62b370e2fb90203ac2acf602e227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_449ff40588639872a2f9957ff2063b34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbfe58a436f8006cd3d861920b200b11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0a23c0e7dc7a3a3306ecfd8daafcf87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c310041ad5b906560b8bf7984b812053
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe0bdd8631c46c5e9f4b091d800e4275(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_224b04c8c71454614df44fc0973961f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 576, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b318facc27f89dd0edf556633021a3ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_224b04c8c71454614df44fc0973961f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_073afdef82ed2fa65c55b9167e4e54af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c8bcabcd402a65525736ae588bfb1a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b26fc58f25cbd0ac0159fd15f74bbfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5b1e5b925959ee04f510dd1de764b95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e02d184efd8c44aff21e51047c8909e
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3094db67e6427a16cf940072eaeef4e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3563b110840540cf19b01b4665ff4194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3094db67e6427a16cf940072eaeef4e7
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1192f0c96a780df9d800f10109adb625(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01b2df4c8acb1765e32dccf862a00b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9acb30993fa3d7f568459a5ffe80ea2
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_018c28c6db1828c1248151c4caca5be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_805deca12a4e89db1d64eea7e0a5248f
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3259a8119e981c5b6740f2d460bfb285(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 288, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7c8b12074784da3425031f5da3e75261(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3259a8119e981c5b6740f2d460bfb285
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eec1e6278095e36d2bd419cdae772ddd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9acb30993fa3d7f568459a5ffe80ea2
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_38ec8ecb531c44ceefeab2949b93c538(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a40ce7f0f872642139e16f5782446d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38ec8ecb531c44ceefeab2949b93c538
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b257955bf4a6063f4e97d0790dc3c88a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c94ab3200a3bd5a6ee8671630b4bcb5e
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b257955bf4a6063f4e97d0790dc3c88a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c94ab3200a3bd5a6ee8671630b4bcb5e
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_224d3bd4104ee8e5cbf9663996b4abda(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 7, 10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3857777f702044ad216b09fd2035d55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_224d3bd4104ee8e5cbf9663996b4abda
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e793dbb84ff36f6889f23886c63b099e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fc42873fe258374c89b096578e41639a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e793dbb84ff36f6889f23886c63b099e
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_162319e0a5b090cbeebf2727e7b3e05c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f6af331bc22684b9c4bb6dfcb43417b
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8a0eb3b641439785badb91a22dccd52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a589eedcb25a0d94cecb4a372edebba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdf7d495bee27942e2dc21dc722a6732
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58ffa96b5cac1b346d18e38a854299b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd596c459fef3c9b64857e206e5b7399(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_82a3d454ec6a0f964a0f96560ce84050(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle.nn.functional.sigmoid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 100, 152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2aad44f4c77e96bccae909a069a9f6db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_82a3d454ec6a0f964a0f96560ce84050
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_caee6dddffc9796c44291a81d9b12383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe8d67bef90555fc81e8965cb8299b22
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca0ba18922ce0dd0aeac740a42847ef6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae0b456a4bab379079bccd314ad96928
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd4b2cae00166cb134348f1795b763e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_922c01e541e70d8dd9a106d34db77822(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96694bac3f8931e01866af465484da6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ef96babd87221c2dd52db48c2be1138(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3092188f29a6dcbc421f39e7e756050c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1e89d13e273b9f8ebfcf9f0bc57b3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d721281f7f446b2372c521e2e2a81ef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58ffa96b5cac1b346d18e38a854299b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de945c1e033a3aaebcd5eeb62607c1ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd880d1e9e39721be50e486875773de8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea479b2fb96546cc7e56632df75a8ba9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 112, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7423c3376ed0d689c635f8b05d1c16ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.212127923965454]], [[1.789625644683838]], [[1.933072566986084]], [[2.306485652923584]], [[1.1387118101119995]], [[2.362990140914917]], [[2.7894678115844727]], [[2.110621690750122]], [[1.9468934535980225]], [[1.6261770725250244]], [[2.458838939666748]], [[1.5788171291351318]], [[2.1331140995025635]], [[1.3116607666015625]], [[2.6825926303863525]], [[1.6499090194702148]], [[3.0373640060424805]], [[2.0838427543640137]], [[1.774763822555542]], [[1.7819828987121582]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_734253be9b81a58c51b5ccfddc9725ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9a2b5bb12a0f08219c8721b9b9ddb4ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2f06854817eecaceaf98de639f1ffe4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55945300ab05b1a81f01ceecef6fb0b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dba9bdd864647b40fdefcfa661c7f6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24471a510fc64e2e28874eb33e01ae56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96694bac3f8931e01866af465484da6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e85e9984f32970a1b9b146234b5a574d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55945300ab05b1a81f01ceecef6fb0b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afd291ec6bdd122574efb8148b38cf9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b6b9ab6c3e4e77d04554b5830151edf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([150, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb92727a25d743927f7520ebfb6b8d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20ce7319a693de1f39188ad52b31a4f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3b65d26b96c63a675ace3f71bd38703(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3139b59773e556e51dc8c907eee38aff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([40, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78b3162838f7f8e0572f39cdb8dc04bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e1f1b81f0b991ad077af7d7e86045d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8df74dfb3c454a6b1403e8e8f7d0ede(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f75bf00ca9a68cdbed0a739e8b54ac69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be0cd4f8c2d1d0b301daf54e7b3f995c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45a265cf30a0d0e1dbcb77dedf0e546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3f75e2631425f035f9f68c63d175d44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03a16a6a6d8bf23f7aa8b6c5fee8292b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f795edd1d030a261be7723256147673c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d721281f7f446b2372c521e2e2a81ef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c9870beeec61c0efddc845d9c913cce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96694bac3f8931e01866af465484da6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9463c504151e77cba49829843806258(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80c3a08495da1ef8bb9f3d8e1781fc22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe374a0ce5f70cd2b7b9f46baed87f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e85e9984f32970a1b9b146234b5a574d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24471a510fc64e2e28874eb33e01ae56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be0cd4f8c2d1d0b301daf54e7b3f995c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65750189b9ee00391ddd8f88320547fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efd6e36b09299dfb2c7a92e65183019d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ca15c345ef7da446407850078e5b961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a78dec507a953a3c9c5241b162004589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94cf39d6a43ca06cc09b1340acd911b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a70ab55891eccf1e301d93d21f7edb67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275ad47bdac9febafabbb5980cbd6ad4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a78dec507a953a3c9c5241b162004589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d0388a06c13787855a5f5f5e602e699(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58ffa96b5cac1b346d18e38a854299b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aaa7d37e281eeca2ea888f2e24a25b00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3f75e2631425f035f9f68c63d175d44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_75c15f2c13a0dd5a09f9185cbdfc0cf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7015325d9f754b309072e52f1e91d8e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f7c15b0e86070229e344cd98c8a7925(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([15200, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb7027f600b3b2248fecca41ed8c7848(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c95694eb664c20a5c9b18994403cd35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65f6f1fd5dba7ec86325d139b20d9209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f84f50a9c2fedc8344c1d2dd758e94b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4640ad20b6e7628608ea2669b318ccb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51d791455990b6f66fb97903fe63a467(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55945300ab05b1a81f01ceecef6fb0b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31463778049a5ebd552fbceb6fb6708e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24471a510fc64e2e28874eb33e01ae56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_70702e66f9249b269e550c165e9d7c4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9339c7c80f077142833a82425eb7d063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62e245202532d4237d8a3163602c81a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94cf39d6a43ca06cc09b1340acd911b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e85e9984f32970a1b9b146234b5a574d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f75bf00ca9a68cdbed0a739e8b54ac69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18a69fe0ba2e8d93989a77bafd79ecaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bab3267dde49bc709480a99e6181195(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc11b487997c6fa970cd426ecaddf35b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f84f50a9c2fedc8344c1d2dd758e94b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9359aadcaf3b84687135fb5077077d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d73b5c1ca459993968fe92deae7ed81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fac61b294bde835368ed1ada3ae604de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af57b2ebe21fbda1fddde37ce123eb44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 13, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03027e005a4a9aac15a14c0fe8ab2fb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f153e0bf2c755aa26f4cc39009a5e91a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([2204, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd4b2cae00166cb134348f1795b763e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62e245202532d4237d8a3163602c81a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fac61b294bde835368ed1ada3ae604de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9339c7c80f077142833a82425eb7d063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd4b2cae00166cb134348f1795b763e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c812ad784c0c6e3088ca11f8bf04b6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([551, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc34da914c49024c46776b0e71772026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1528cab2de06fcc6d3489e2b533127bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fac61b294bde835368ed1ada3ae604de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4c5ddc74894f155651585d12c8e71597(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5396bfcd914b183ed29797404f56769(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3560532ff7214a5ed9cb031067fda273(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([247, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fcacef14c09722b1c06548ad5ddf8c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([950, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e06a9a9b2f05ad6fe4fc38dc9eb80b7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_53985539acb423cf71b56af59f3ff2fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e1f1b81f0b991ad077af7d7e86045d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9af5b8a8ab48f7917d64c7a914f0267(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65f6f1fd5dba7ec86325d139b20d9209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fe5beb9d255699f3a58442eb9fad2ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd4b2cae00166cb134348f1795b763e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc34da914c49024c46776b0e71772026(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e514c338ecd9c49d7a401b54d312298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be0cd4f8c2d1d0b301daf54e7b3f995c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb92727a25d743927f7520ebfb6b8d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b349fef70e301c380190099f2e33537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24471a510fc64e2e28874eb33e01ae56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45a265cf30a0d0e1dbcb77dedf0e546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45a265cf30a0d0e1dbcb77dedf0e546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0c66baebbffe0173b4e67bb1f17c3d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([8816, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_275ad47bdac9febafabbb5980cbd6ad4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cc8a25731ea6fdf0aacb9adf6c0fa94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebc725b4ec959ef4c24d6b67644ffccc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd4b2cae00166cb134348f1795b763e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65f6f1fd5dba7ec86325d139b20d9209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efd6e36b09299dfb2c7a92e65183019d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3b65d26b96c63a675ace3f71bd38703(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18a69fe0ba2e8d93989a77bafd79ecaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f75bf00ca9a68cdbed0a739e8b54ac69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9339c7c80f077142833a82425eb7d063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfa0f168c53b0853f9e4fc7621cbedb2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16cc225448d83d7fe904f377918cb380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e217968ada64d3a769c0101ccb613c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe5337e8b74fa4567796ba763152588d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b5fd536332f0ba6ea252197f9a2cc3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb92727a25d743927f7520ebfb6b8d12(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8a0eb3b641439785badb91a22dccd52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f5e97c61a630f7448ccf4dda6b92589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe5337e8b74fa4567796ba763152588d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_96694bac3f8931e01866af465484da6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55945300ab05b1a81f01ceecef6fb0b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_659d785d28ddd39b32b62bfa2021a388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4571b63363b1db383087d0774f800fd4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2a3cdd57b2386031e1895fe4732cf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a3fe9ab6e6e04cbc954a735897f9de3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_717f62e10669f8ffa05621258e92eb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_659d785d28ddd39b32b62bfa2021a388(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0a63e0213c6cfd45c999bbe1a22fb8f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d73b5c1ca459993968fe92deae7ed81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1528cab2de06fcc6d3489e2b533127bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d721281f7f446b2372c521e2e2a81ef1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac3b8587724db1995331d1302de4a1cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.7184655666351318]], [[1.7496981620788574]], [[2.094937324523926]], [[2.0464823246002197]], [[2.109078884124756]], [[1.7068365812301636]], [[1.2507412433624268]], [[2.1731138229370117]], [[1.676121711730957]], [[1.1951981782913208]], [[2.1150741577148438]], [[1.6924262046813965]], [[1.820841908454895]], [[0.9820704460144043]], [[2.175051212310791]], [[2.179396867752075]], [[1.4661563634872437]], [[1.5999906063079834]], [[1.7346348762512207]], [[2.3203139305114746]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_734253be9b81a58c51b5ccfddc9725ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7970b3aca65cba38b7afff9b415013a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec1813f7f36dcbddf24c320bda0dafe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4458b19e9a4df105f97e76373aff76c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdf79f661d1734065543868594d644f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee906f51840bd6ba9d8b184f979636bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62e245202532d4237d8a3163602c81a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e1f1b81f0b991ad077af7d7e86045d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45a265cf30a0d0e1dbcb77dedf0e546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cb1f68fc62fa8911510188e06b9fef0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1ed58c33699f24e876587d215dc0202(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 120, 200], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65750189b9ee00391ddd8f88320547fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_055471b20041bee156c7542aae30a9da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6d2fa10fca2c68a739dc5577027c4de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff1d9581fc711bd204607ac99f72155d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a70ab55891eccf1e301d93d21f7edb67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d45a265cf30a0d0e1dbcb77dedf0e546(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55fc8cc9a5bbbd364538d502556ebf92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[2.7839205265045166]], [[2.3346548080444336]], [[2.28593373298645]], [[2.7708616256713867]], [[2.30279278755188]], [[3.2229268550872803]], [[2.696528911590576]], [[2.943495988845825]], [[3.1310055255889893]], [[1.8862913846969604]], [[2.6206557750701904]], [[2.277665853500366]], [[1.9456156492233276]], [[1.5896580219268799]], [[2.2188262939453125]], [[2.2632229328155518]], [[1.6459977626800537]], [[3.8634533882141113]], [[1.8549753427505493]], [[3.404752492904663]]]], dtype='float32').reshape([1, 20, 1, 1]),
            ]


    class TestPrimitiveOp_734253be9b81a58c51b5ccfddc9725ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7970b3aca65cba38b7afff9b415013a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdf79f661d1734065543868594d644f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dba9bdd864647b40fdefcfa661c7f6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([3800, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62e245202532d4237d8a3163602c81a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65f6f1fd5dba7ec86325d139b20d9209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d73b5c1ca459993968fe92deae7ed81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24471a510fc64e2e28874eb33e01ae56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b71e62b370e2fb90203ac2acf602e227(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_449ff40588639872a2f9957ff2063b34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbfe58a436f8006cd3d861920b200b11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9359aadcaf3b84687135fb5077077d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a632cadce3ad7421a2594958b4fade48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b734d3d5aeefb7c48923d7adc63ab151(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_073afdef82ed2fa65c55b9167e4e54af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a70ab55891eccf1e301d93d21f7edb67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b26fc58f25cbd0ac0159fd15f74bbfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2dad4154baf15c9bdd0432f52d81562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8764ec62403792b37f34ad20d575cd46
        def get_inputs(self):
            return [
                paddle.uniform([70, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4740b0cfae67c941d7d587c5aee69b2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20ce7319a693de1f39188ad52b31a4f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9af5b8a8ab48f7917d64c7a914f0267(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ca15c345ef7da446407850078e5b961(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f577f9557198a3461302632d51adf2ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bab3267dde49bc709480a99e6181195(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6776e24e8a5cdeb2cece6cb6826230a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1528cab2de06fcc6d3489e2b533127bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1528cab2de06fcc6d3489e2b533127bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2a28d04a5f84b3a6764c6525b4375da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 7, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a68a11c7b645e132895a9202fce2e5ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7c8504f63a36b37301ccec64a7f45fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 9, 14, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b8a0eb3b641439785badb91a22dccd52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9339c7c80f077142833a82425eb7d063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_58ffa96b5cac1b346d18e38a854299b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd596c459fef3c9b64857e206e5b7399(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b648aa4091a46f2a093edd14b79f02e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afd291ec6bdd122574efb8148b38cf9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8dede05ad7074b508784b069cf67eef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e65ccf175dfb7b637b72fbec7bbc3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 192, 288], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()