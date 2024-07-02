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
    class PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68c928605a3b6137872b660bb25694e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f14e3b36193142315b066449bb5333f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 91, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_984eaa668012cd8e64cefaaf3920d34c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_655d12020926be4c828b05590e2f36ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0823831017cd6e2a49f4f56875c1de9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_655d12020926be4c828b05590e2f36ad
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_37068a43619314911c77b9dc82a2362f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9127be9303bf1af01901b97598b8cd40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([512, 256, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b7d43c138eb2841cc20fbf6077efd1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 68, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bf5a3977da623fe5d81eedd91574904a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18d07a66e874298ebb5d11aedbe6c231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81ddeb4713b50ab7780185bfa7a3f494(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_399ff42015e1fee1e8e43986c10dd6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_971434b1d1259366713a05719b599893(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1fd3d180350fd99c5b72a110653859(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020334da83088182891e5a0e6a084e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_801b946b7adc5e9ea1b837c2a79a7160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0370c69d10ecd841a08e1cda11ab0c4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2db4bba64414053484e635bf30b999c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5e7a302ee7e24d8222aba89d6373b229(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 0, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_616f43c25f1569edb5417990036d8387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e7a302ee7e24d8222aba89d6373b229
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    
    class PrimitiveOp_14eb8e953421f1d4e152937eaf8dca25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8846212ff5609898908df053fa751046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14eb8e953421f1d4e152937eaf8dca25
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int64'),
            ]


    class TestPrimitiveOp_4931fead6bdbf3e9de38e52c1023893d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40587de250b2ee389e175cf3ca91328f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_22ec1bd90f5c8c4b2e00a929f7b26958(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_741f132225899beb39bad155fc40e8d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22ec1bd90f5c8c4b2e00a929f7b26958
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1b80173f74cdf2d52d46ef25bc30bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6e53492af60d923fd9a2560c1ba4a8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f10bb5230aef071c5ff6a9f11f79279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9068a45d3c4f43781c0f7842ad8dc3e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 76, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca3ff8779cc80f258ef484fef49fae8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9068a45d3c4f43781c0f7842ad8dc3e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0bd9a12fd44bb4b12e12aa80ab8f8efe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fab53191a84c864d63cef2997a13e4bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bd9a12fd44bb4b12e12aa80ab8f8efe
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_44755e74b1f2b64d9c9215c46762ae85(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1000, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_065141395872dabe3b47548e13117cb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44755e74b1f2b64d9c9215c46762ae85
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc64d0e7bf4c038ae9984a792c379974(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdac7e5db4d271e044774c0a61a7db8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b23b5bb3245abea4d2790425d331bd69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c41e44ec5bbc84073c51983bb4ec7ae1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6111871c8fe83b57f861a760c72983d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 15, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6855bf3d0558d8324edee731ac16f15e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6111871c8fe83b57f861a760c72983d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600a05a79ec6bbd1d743d75a45915eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d0f05e60fe429a28a5a02aee25327c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9068a45d3c4f43781c0f7842ad8dc3e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850c1eeae569985b821b627f27907744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe09d75d480fc8a91a916cd6030e998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4f440f112a259a576e3dfb7bbc5180a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8f647166549d22cfbb4a97ebb16ebfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbd4f408860fa7b3cc8e6fe01525a689(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e5fbb35b4f1267048c42cacbc3df1a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f1f6e6423d6fc7e31050326f83edd84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9d09ce248b82a2ab5b7236182375d17b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3a9e73713fa551b382bc39850f038599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d09ce248b82a2ab5b7236182375d17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_971434b1d1259366713a05719b599893(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_787033f7406baa11939cf701a9f29f7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ad00eb583a5dc95b80b80e8e62b1652(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c275cfe9b5eecf8bef49f1cd7996a1a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22ec1bd90f5c8c4b2e00a929f7b26958
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_131da882a17a256a6776eb02be5d2cfc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6f239480fd8bf7aed38f94964a3a92d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_131da882a17a256a6776eb02be5d2cfc
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdcbf557b184cb1328815581e99389da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1142cd87a998853c49535df677f621e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38426eecd86ea28388f6836ba25eb383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_755e0b689ed71e587b0d89dbb98eb36d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af874d8b12cd3d4a28518928fca5818e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_755e0b689ed71e587b0d89dbb98eb36d
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_091e2f5b81e065ad09ec64e5e4e0a791(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6111871c8fe83b57f861a760c72983d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dbf40d8fd715a7f2d01a2bb5de0a380(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6111871c8fe83b57f861a760c72983d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c088bac31b0bac25b38a9ed9184e112b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_655d12020926be4c828b05590e2f36ad
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600a05a79ec6bbd1d743d75a45915eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90ec534f09a50e96538b36f2aaf05200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9d77a48b8677dac912e5cd063c70e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([390, 64, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_585343d893aa8f0aebdc4fe61887a5be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ac819b52d1593e1493fecf619a67efe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b94509319a86278de84332619d957e67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_131da882a17a256a6776eb02be5d2cfc
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d7ec041dda05746935a810b15be08fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 1, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_795a5c67dc49a344ef5aaa02aa8704c9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d7ec041dda05746935a810b15be08fd
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_da7c679d1990748316c83b550a052439(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_417ce2f73545ada5928b410013cc1b0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da7c679d1990748316c83b550a052439
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_be7ada960bd66aa4a14e23792df765df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5e4f9912004e0cc66df8e964b78dbea0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be7ada960bd66aa4a14e23792df765df
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1c9015aa06c3171739c54c4db1a09f32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 200, 304], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_003b0a3d188ab2b77e8ba4e9256c716e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c9015aa06c3171739c54c4db1a09f32
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9950de879da6ccf7b63a594913c94a56(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 5, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7017dcaa1ebba06263d219d0a1d2a184(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9950de879da6ccf7b63a594913c94a56
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_412bcc39a10bf9c702ab30b8e28b8f8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04c22b515011de5ede8efbe183fa62af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cff51646650728e2517a5e77289bba8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_309fcc06ea1e27f187486d5e4afd0455(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020334da83088182891e5a0e6a084e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_801b946b7adc5e9ea1b837c2a79a7160(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5deee4494208e7e440225e197035b9a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2f9d3cedc7568551630e24c4610fe2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8f647166549d22cfbb4a97ebb16ebfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed56c5864533a3064fe4b907047ce2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0a31f29994879dcf76082d5809a3c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_60bd48fccbab44ba8d9879469b909dd7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1280, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a30a30636a468503d637f1d64a1bf7e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60bd48fccbab44ba8d9879469b909dd7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0b6c802671b08cd6e493d1f1dd151aee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5bf3ca33186f253165f3b0080fa9fdd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b6c802671b08cd6e493d1f1dd151aee
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3c35dbaa2f35591ff968dd20771d816(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bc3490371208a1b0b335dac69f07dcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0824c8da8d6101a0a08d9e3d52cfbeef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3ef89239b6f8aaf893cb5faa8d871fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bd9a12fd44bb4b12e12aa80ab8f8efe
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d76088b80a21109349cb83de3e591c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dc77ab335eae088691ba75d3b168e53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26a7bd67b3d4cf284334e0bd529dbafb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_94093d285322a0974b4540e4a30eb3f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63661b1427711650c792f73f2020f144(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2fc7fe26e99aa294f315cd78807c94a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be7ada960bd66aa4a14e23792df765df
        def get_inputs(self):
            return [
                paddle.uniform([6, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf1bcebf2f145c807dd01fc3ee6f990(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48d063e46c268192d2529c64f2369e76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1b80173f74cdf2d52d46ef25bc30bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6e53492af60d923fd9a2560c1ba4a8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85449d9d91838e39fdd02f0c356a320a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_309fcc06ea1e27f187486d5e4afd0455(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb6cba2382303a15dcb2588ea65d7cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c11cd204b15acd27c70f62a7dbc6978(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d76088b80a21109349cb83de3e591c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8ceb969ab783c674320b3933964aa63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_664d2726a2fdf18fc8cd535364f33e25(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 192, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eea8254eb536cbd9d9b99133ebce0e0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_664d2726a2fdf18fc8cd535364f33e25
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7c635f05c336c1de3c0c6c4c75af96c3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 384, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_281b71454acc98ae032165c9e4c6b3de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7c635f05c336c1de3c0c6c4c75af96c3
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_37e75eee63849f0f379fab0b00d67be5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3fe09cc1194db58b5d4a903d05ffe541(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37e75eee63849f0f379fab0b00d67be5
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c148e8d6e6301ca53278f45bb0f909d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e7a302ee7e24d8222aba89d6373b229
        def get_inputs(self):
            return [
                paddle.to_tensor([[[6], [6]]], dtype='int32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_02da62c1e9b88156f52fa0fc488813c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14eb8e953421f1d4e152937eaf8dca25
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int64'),
            ]


    class TestPrimitiveOp_3a9e73713fa551b382bc39850f038599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d09ce248b82a2ab5b7236182375d17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34706dbd3eebafdf40cbec716b69889c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 1, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_13e5ad927abfafd24ad3bb7b089fefd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34706dbd3eebafdf40cbec716b69889c
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 1, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_866048220c9c213adc30b2c185fe853e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a87014374f2ff3d965404a35de49b694(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68c928605a3b6137872b660bb25694e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_984eaa668012cd8e64cefaaf3920d34c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a9e73713fa551b382bc39850f038599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d09ce248b82a2ab5b7236182375d17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ab57429c5d01e13fce6e8a082d1cf76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be7ada960bd66aa4a14e23792df765df
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5d5e40a2c424356b3f74c1ba083ac14f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 2, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb933fd506f3675ab86ecc660f099b83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d5e40a2c424356b3f74c1ba083ac14f
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68c928605a3b6137872b660bb25694e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_984eaa668012cd8e64cefaaf3920d34c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_53f268975b70db40c51446102ff5803c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 96, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_771f74894ba40bdcb5cb545d096626c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_53f268975b70db40c51446102ff5803c
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_616f43c25f1569edb5417990036d8387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e7a302ee7e24d8222aba89d6373b229
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_9dabe728505f92e8249716b85f935093(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14eb8e953421f1d4e152937eaf8dca25
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int64'),
            ]


    class TestPrimitiveOp_0f1b9c90c50013bc07103a2abad813e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7deec4dbfc215c258b64eae6ab9893b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5424223cff13d69958d855be9f9f37ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8670125ff6280e9f5637b1f5310e4537(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7669357f94c55631c4fb90ed8b07dcad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_755e0b689ed71e587b0d89dbb98eb36d
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_86b3961f0fc9d407afadb19c3042bacc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, 4, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d47e15937a12de97e07f7a9be50beba6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_86b3961f0fc9d407afadb19c3042bacc
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30501c124e1360f7a97384c5e5fa72e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48d063e46c268192d2529c64f2369e76(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22fdedcd6d8c713e6b788f80c475df5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0bd9a12fd44bb4b12e12aa80ab8f8efe
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7471152917d37fe986848225742dc40a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b14e0adf223b660b5a9b68580d4caa10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cbdf4c2375fcc5a778ac48738bcb7a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b23b5bb3245abea4d2790425d331bd69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0b0852897e33644ea5d6c99f03bbd77e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9068a45d3c4f43781c0f7842ad8dc3e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a829ef6446b854c075be82dcd3291568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_44755e74b1f2b64d9c9215c46762ae85
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_583e5398b05ba32195e693db433c0466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b6c802671b08cd6e493d1f1dd151aee
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49d9dc0606e120ce59443a32555abfe6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, 4, 80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb0f33d7806dd395819898da3e8081b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49d9dc0606e120ce59443a32555abfe6
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30bf83ee85153a1d08371643b1b7e06f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6111871c8fe83b57f861a760c72983d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f1b9c90c50013bc07103a2abad813e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f54c6613a1bd5222e62bc5e053501f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f22ddd21683ed2d460068ddd81079a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600a05a79ec6bbd1d743d75a45915eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90ec534f09a50e96538b36f2aaf05200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85ba9e2ef015a50123eb1002a2273f05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da7c679d1990748316c83b550a052439
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f10bb5230aef071c5ff6a9f11f79279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e08b7fc37bfcf8a0e04622e1f104e519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5c4cca496a208967e91b2fd9800628e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59df1bf230a7a7a2c7a5d48ec93e1890(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1b80173f74cdf2d52d46ef25bc30bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6e53492af60d923fd9a2560c1ba4a8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_39a27e75d7d8d42d136eafda2aa133a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f26e9776c6a29d55d4ea0b5703152958(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39a27e75d7d8d42d136eafda2aa133a4
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850c1eeae569985b821b627f27907744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe09d75d480fc8a91a916cd6030e998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4f440f112a259a576e3dfb7bbc5180a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850c1eeae569985b821b627f27907744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe09d75d480fc8a91a916cd6030e998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4f440f112a259a576e3dfb7bbc5180a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4283467453b14cc857229196b17ab978(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24, 24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_72bb7471e59e81b403e6ff307d6ba0a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4283467453b14cc857229196b17ab978
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72bb7471e59e81b403e6ff307d6ba0a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4283467453b14cc857229196b17ab978
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_412bcc39a10bf9c702ab30b8e28b8f8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c58e336b2d3c40a73a0f7ac467a45956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4ac4e9c477d58c64f8fdb18b5393fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c25ff725790a918b463369b4428917f9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 384, 14, 14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4a6a0975393fc8beba9edb1b870df066(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c25ff725790a918b463369b4428917f9
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_508dc86675871ac2ed515dcca9dcd765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9950de879da6ccf7b63a594913c94a56
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_584bbeb17bbe5bf59338b3bbd5425fe6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_79a14fa6f029d28281d03045462460dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68c928605a3b6137872b660bb25694e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_984eaa668012cd8e64cefaaf3920d34c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a4b7201c609c0d4f40c4ce622b3c2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d66f8c0f225f64b43448a98482786c2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf51ee503922d0ae965a0c7bc491dae8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 100, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a652624f936e5678a6f2748f47dd2a61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf51ee503922d0ae965a0c7bc491dae8
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01e3dff003c69ef1deae934ec8800c39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d7ec041dda05746935a810b15be08fd
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_93ccde60d9cdbd74d0f281fb26f0e52b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, 2, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f46364be63235541cb136e71bc2b9d88(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93ccde60d9cdbd74d0f281fb26f0e52b
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ae1e0c66049515de5690954130fa5dd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be7ada960bd66aa4a14e23792df765df
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb5cd0da6f39a46f44d556656f5fa114(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, 136, 160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9491a44171ffc33fd1abe35cc12da07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb5cd0da6f39a46f44d556656f5fa114
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a9e73713fa551b382bc39850f038599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d09ce248b82a2ab5b7236182375d17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f1ced7f3e21dbc0f3635dcd14bcbb4eb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 300, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7370994423573c275f349999ebcc1683(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f1ced7f3e21dbc0f3635dcd14bcbb4eb
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f10bb5230aef071c5ff6a9f11f79279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e08b7fc37bfcf8a0e04622e1f104e519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2ee72f9c7f9ec81888cf84cc537553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd934f39bf95bced142d14fbedc08be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_840f3caadc712cacefd649bbe7012bf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63b319dd447aa768cbdfb22d2eda413c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f932c9d3d9ffc0993ce9931ddca124a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04c22b515011de5ede8efbe183fa62af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2c533e20afbd30c727c697f85ff2e96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34124d9a4c3f69dc0ac1693e34f84b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8198ca8d874c4376049e78649d9ac96f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b522f68086b0c48059227146932fd71b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9950de879da6ccf7b63a594913c94a56
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d420c2e53a93d04b183d22c62ec8858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92c350572ede86fabde5c0d1e14154a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_57695f696d9625a60c92f401670fa7e3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 768, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a89238910f0121d25058286cc8296879(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_57695f696d9625a60c92f401670fa7e3
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f45bdf2ddf153d235a6f3950f5fb16d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be7ada960bd66aa4a14e23792df765df
        def get_inputs(self):
            return [
                paddle.uniform([4, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9848e8dff9606cffd537a985dec7d02f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ac819b52d1593e1493fecf619a67efe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4eb1a80e798e756ffdeda7b9a9f13163(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da7c679d1990748316c83b550a052439
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_971434b1d1259366713a05719b599893(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1fd3d180350fd99c5b72a110653859(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e696ae82a74a8813fec4814094d960d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9950de879da6ccf7b63a594913c94a56
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_374ab4e0dd0f3d707c9b957806b8d081(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 384, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28806bc2a253517ed8e42a34d3b4ee74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_374ab4e0dd0f3d707c9b957806b8d081
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6d82ea83d750012a7b901f60ad4a2327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60bd48fccbab44ba8d9879469b909dd7
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3bd217832644aa923e4713cf0ebaf461(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 768, 7, 7], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_982753b0af39507c82b8906d0cfe96ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3bd217832644aa923e4713cf0ebaf461
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b23b5bb3245abea4d2790425d331bd69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c41e44ec5bbc84073c51983bb4ec7ae1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97a61c9da37a6867fad7a0a206f30892(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6111871c8fe83b57f861a760c72983d4
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_585343d893aa8f0aebdc4fe61887a5be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8fa5d53f4b60c330cf4c3f6d1e4a315(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01826525dae847acb4b1152cc0e0cac8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c770c4e1f6e4a3e29e3cdd2af80197b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fc395005c30671a434b95de3f07d79f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850c1eeae569985b821b627f27907744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe09d75d480fc8a91a916cd6030e998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4f440f112a259a576e3dfb7bbc5180a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1b80173f74cdf2d52d46ef25bc30bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6e53492af60d923fd9a2560c1ba4a8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f14e3b36193142315b066449bb5333f2
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da64ea60645fbae49f09359fbe144294(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a58fbab33aa63b2df537347790df67c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e54512e41bc31b8770f430ba4513c345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2a58fbab33aa63b2df537347790df67c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_052a234a8e0ee438fe0fdd38d37f8d1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_374ab4e0dd0f3d707c9b957806b8d081
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55007aae5da79e3ea3876865436c5ec9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc394c46de8c2f031a299bcd9872e48b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16a373bd0ad5c0be6209d92e32f9d7df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37e75eee63849f0f379fab0b00d67be5
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f91c23913fb890ec120bcb90850e257(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a08138573f6728be27a9302dd515eb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9ddba86302471b7848b36ca4a7620360(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 192, 28, 28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5fc59e45a99f854fc722f94955425fe5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9ddba86302471b7848b36ca4a7620360
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c90fd671566359d33f8b9faed7e89fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d48fa85a8cb832b03c5c158633388a15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c39cbc223da03a2d81d32e6e99bc2bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f476593d9c469fec2ce92e51313ba686(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c39cbc223da03a2d81d32e6e99bc2bf
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33998f086daf6288c6d88b71e0f82dfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c39cbc223da03a2d81d32e6e99bc2bf
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a4496b6c95b5cab323198ec8c7c95168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9950de879da6ccf7b63a594913c94a56
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2abd84f3c5736792efc3fc77869305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39a27e75d7d8d42d136eafda2aa133a4
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2ee72f9c7f9ec81888cf84cc537553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_798d3d1c9dc32be35f8f54ca5e9a1f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc1056c0fdf0c70a7b756df12a5cae37
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020334da83088182891e5a0e6a084e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb15081c00c457334dc0f5f82f250029(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fb394cdc81ac047938c45a2f1a44df4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1144dbe4e8205fa1216e05db7bd06fcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bb42dd1bcab193f3912ad2db635412e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_be7ada960bd66aa4a14e23792df765df
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_733a68d415daa5c3002bfa8a43274131(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 2, 3), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 96, 56, 56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b72c9010822192ba9a7e59caae34d173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_733a68d415daa5c3002bfa8a43274131
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68c928605a3b6137872b660bb25694e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3639c5389c9c4d785246e1b41bd7e079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd9446ed97ad7c332f7ecdde830d060b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9127be9303bf1af01901b97598b8cd40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([512, 256, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b7d43c138eb2841cc20fbf6077efd1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3e00d1aec9551bd2f050aebf85e9559(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 84, 84], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_18d07a66e874298ebb5d11aedbe6c231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81ddeb4713b50ab7780185bfa7a3f494(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_399ff42015e1fee1e8e43986c10dd6f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_971434b1d1259366713a05719b599893(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_767136b47697554ccb5d348991a12809(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020334da83088182891e5a0e6a084e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c37ada5554bc72fd63c1f9e9205140cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0370c69d10ecd841a08e1cda11ab0c4d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a278ffd65c4a5ebf2f7751c6eb1290eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 15, 15], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_616f43c25f1569edb5417990036d8387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e7a302ee7e24d8222aba89d6373b229
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_8846212ff5609898908df053fa751046(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14eb8e953421f1d4e152937eaf8dca25
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int64'),
            ]


    class TestPrimitiveOp_4931fead6bdbf3e9de38e52c1023893d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40587de250b2ee389e175cf3ca91328f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9112131950c9ef759be2e7fcbb53ba3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1b80173f74cdf2d52d46ef25bc30bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c3d0dfa8611ec4fdb5902de31a39d73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f10bb5230aef071c5ff6a9f11f79279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b21c1271584c55c69eab7ef06e6a4a01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9e8708276ad4235534d9c02ff9c8ebc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([11, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f88bae97a30d4a12d8ea753e4028f76d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([10, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc64d0e7bf4c038ae9984a792c379974(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cc08167446c33531a0fc3f653d9f6b97(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b23b5bb3245abea4d2790425d331bd69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1983b3e9534069e6bc82798d19f38015(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_715496c9e8502253d97180cf5c5594e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600a05a79ec6bbd1d743d75a45915eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_014344385248caa4899fc26d89894b2c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850c1eeae569985b821b627f27907744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe09d75d480fc8a91a916cd6030e998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_985be251483665daa6648aad2019aec4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8f647166549d22cfbb4a97ebb16ebfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f47a6c4a0364a47c2c4b5a7b12c52beb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e5fbb35b4f1267048c42cacbc3df1a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b04a039e148ba8b4d2b6cfe951181bff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 76, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a9e73713fa551b382bc39850f038599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d09ce248b82a2ab5b7236182375d17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_971434b1d1259366713a05719b599893(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_787033f7406baa11939cf701a9f29f7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ad00eb583a5dc95b80b80e8e62b1652(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6f3fb541a3002cb7fcca0a941867282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca60bbd7d03ba0b6ecd672f6405ec562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cdcbf557b184cb1328815581e99389da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1142cd87a998853c49535df677f621e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38426eecd86ea28388f6836ba25eb383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 36, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9ef3414e1c9735aa1d062ffc02fa2f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_165882ceef7c6b43d4236978e367d96b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1b80173f74cdf2d52d46ef25bc30bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_296c7dd9dbcb005b0e4d9017eb8d79e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600a05a79ec6bbd1d743d75a45915eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be09975fba2e627b5948c8a58aacdf65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9d77a48b8677dac912e5cd063c70e9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([390, 64, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_585343d893aa8f0aebdc4fe61887a5be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a29428cd9939fed82f7390bef076a553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e6cb18772f30308224d42408bc5b3f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_812b8ade1761acab5cf7868a60536e77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 1, 2), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_55dc4c72d29eb2a81e5bcb08df40de3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_812b8ade1761acab5cf7868a60536e77
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1235175a797f092f5a1776873680d6ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ce1f7e6ef760d72df33f3e83acbfa4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ce1f7e6ef760d72df33f3e83acbfa4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 200, 304], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68b2bdffd06dd3ae655b0d68ff06ecab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_412bcc39a10bf9c702ab30b8e28b8f8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40007e51c14dcd7948d343cddf066fe6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cff51646650728e2517a5e77289bba8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7758d00168e824ea7548d613a9977ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020334da83088182891e5a0e6a084e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c37ada5554bc72fd63c1f9e9205140cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5deee4494208e7e440225e197035b9a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ac4d669ed99b1d1a0a4ab1f42096605(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 21, 21], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8f647166549d22cfbb4a97ebb16ebfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7ed56c5864533a3064fe4b907047ce2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0a31f29994879dcf76082d5809a3c30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 34, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72dfe2d8810a664beb3085862892d794(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2dfbf1d245b9612116ebd15b93fa37f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3c35dbaa2f35591ff968dd20771d816(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7bc3490371208a1b0b335dac69f07dcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0824c8da8d6101a0a08d9e3d52cfbeef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 18, 18], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9ffbe65e777440f3d7bf8595bb2dfdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([43, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d76088b80a21109349cb83de3e591c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5dc77ab335eae088691ba75d3b168e53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_26a7bd67b3d4cf284334e0bd529dbafb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94093d285322a0974b4540e4a30eb3f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63661b1427711650c792f73f2020f144(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfa6b84e4a70d3943a61919679e5499b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([6, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bf1bcebf2f145c807dd01fc3ee6f990(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2eb928c18be827c9ab4f0194aa0e553a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1b80173f74cdf2d52d46ef25bc30bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c3d0dfa8611ec4fdb5902de31a39d73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85449d9d91838e39fdd02f0c356a320a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c7758d00168e824ea7548d613a9977ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb6cba2382303a15dcb2588ea65d7cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c11cd204b15acd27c70f62a7dbc6978(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d76088b80a21109349cb83de3e591c5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ba9238fe2a8e21fc8dac0cb6b5cc2fc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 17, 17], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a3216bfd1a7cb61d41484d1a70e1ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13cc2e40d11de2b71c856b05a378f23f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90d2137641bd40de96538b9fa6660aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c148e8d6e6301ca53278f45bb0f909d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e7a302ee7e24d8222aba89d6373b229
        def get_inputs(self):
            return [
                paddle.to_tensor([[[6], [6]]], dtype='int32').reshape([1, 2, 1]),
            ]


    class TestPrimitiveOp_02da62c1e9b88156f52fa0fc488813c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14eb8e953421f1d4e152937eaf8dca25
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int64'),
            ]


    class TestPrimitiveOp_3a9e73713fa551b382bc39850f038599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d09ce248b82a2ab5b7236182375d17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_06771210a1ea04c7c733b9b9daf6b868(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([22, 1, 1, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_866048220c9c213adc30b2c185fe853e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da3993bd210b1ae4eb3f5e45958cf2ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 46, 46], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68c928605a3b6137872b660bb25694e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3639c5389c9c4d785246e1b41bd7e079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a9e73713fa551b382bc39850f038599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d09ce248b82a2ab5b7236182375d17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1aafb48aae46c6464dce9df9c85c29cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6aa5eee5e5cc4ce096f5d506a9cdc21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68c928605a3b6137872b660bb25694e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3639c5389c9c4d785246e1b41bd7e079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c69120bbcf741bcaefb4cf8f81a7c6e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_616f43c25f1569edb5417990036d8387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5e7a302ee7e24d8222aba89d6373b229
        def get_inputs(self):
            return [
                paddle.to_tensor([[[3]]], dtype='int32').reshape([1, 1, 1]),
            ]


    class TestPrimitiveOp_9dabe728505f92e8249716b85f935093(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14eb8e953421f1d4e152937eaf8dca25
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int64'),
            ]


    class TestPrimitiveOp_0f1b9c90c50013bc07103a2abad813e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4998a3a3a4273da3033b3fad5d51633d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5424223cff13d69958d855be9f9f37ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bee2beec7b38cc1438644467db4060c7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22f60306b6dfb5778f174302813440d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4825442d7d503f31c5d5801ee29eaa0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30501c124e1360f7a97384c5e5fa72e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2eb928c18be827c9ab4f0194aa0e553a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b7aea8b8f677a8db59675cc6885e9195(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7471152917d37fe986848225742dc40a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b14e0adf223b660b5a9b68580d4caa10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0cbdf4c2375fcc5a778ac48738bcb7a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b23b5bb3245abea4d2790425d331bd69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08d03d0693cd712d0d8a4aed13a12943(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 76, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_73f42333eac50e0e3e74dccb12a3c17b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([22, 1000, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6888368585a0c5e82f965bea4f56daf8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23a86f13e6fae5bc77d642fac5586fb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([10, 128, 4, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68c928605a3b6137872b660bb25694e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f1b9c90c50013bc07103a2abad813e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f54c6613a1bd5222e62bc5e053501f51(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f22ddd21683ed2d460068ddd81079a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 80, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_600a05a79ec6bbd1d743d75a45915eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be09975fba2e627b5948c8a58aacdf65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 13, 13], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4eb52ecaf7eee3d01bfcf84c9b7b27b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f10bb5230aef071c5ff6a9f11f79279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6d9c618295111396015b798387d3469(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5c4cca496a208967e91b2fd9800628e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cfee7a59360e279161ed9128df038a37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 23, 23], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1b80173f74cdf2d52d46ef25bc30bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c3d0dfa8611ec4fdb5902de31a39d73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c4f7f30cf1f7e1086099b7a0f90877a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850c1eeae569985b821b627f27907744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe09d75d480fc8a91a916cd6030e998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_985be251483665daa6648aad2019aec4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850c1eeae569985b821b627f27907744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe09d75d480fc8a91a916cd6030e998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_985be251483665daa6648aad2019aec4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba640864d502e897ba8508130839a44c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.flatten(input_0, 0, 1), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6632c9cca60ae98493e5c263c2ee0ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba640864d502e897ba8508130839a44c
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6632c9cca60ae98493e5c263c2ee0ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba640864d502e897ba8508130839a44c
        def get_inputs(self):
            return [
                paddle.uniform([24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_412bcc39a10bf9c702ab30b8e28b8f8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c58e336b2d3c40a73a0f7ac467a45956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b4ac4e9c477d58c64f8fdb18b5393fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cb2acaac2f70a6483d2f6150559d939(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ab1722eab02df387691d576a296b92d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_584bbeb17bbe5bf59338b3bbd5425fe6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a26e2f72ef98f605ae216398124214bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 19, 19], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_68c928605a3b6137872b660bb25694e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3639c5389c9c4d785246e1b41bd7e079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51a4b7201c609c0d4f40c4ce622b3c2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cc3079b004ab064629431054f10e023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d66f8c0f225f64b43448a98482786c2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63661b1427711650c792f73f2020f144(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 100, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10281a6732ea49bdae41d1ff57a24213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_812b8ade1761acab5cf7868a60536e77
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5287811f59c009ee84f8cc24e9fa157e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([10, 256, 2, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_373af2857d07f82584d0f547e7f3eb9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_373af2857d07f82584d0f547e7f3eb9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 136, 160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a9e73713fa551b382bc39850f038599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d09ce248b82a2ab5b7236182375d17b
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c11cd204b15acd27c70f62a7dbc6978(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31123fd58d6aeea0ce2a4bf6cf6369c2
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f10bb5230aef071c5ff6a9f11f79279(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6d9c618295111396015b798387d3469(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 26, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2ee72f9c7f9ec81888cf84cc537553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd934f39bf95bced142d14fbedc08be3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_840f3caadc712cacefd649bbe7012bf2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63b319dd447aa768cbdfb22d2eda413c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([11, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6f932c9d3d9ffc0993ce9931ddca124a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_40007e51c14dcd7948d343cddf066fe6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 40, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f2c533e20afbd30c727c697f85ff2e96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34124d9a4c3f69dc0ac1693e34f84b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8198ca8d874c4376049e78649d9ac96f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 72, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367100a6bfa16fa32b692e5236b86df9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d420c2e53a93d04b183d22c62ec8858(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e65404e5ac95a78f5f8b7895df4feed2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4eb52ecaf7eee3d01bfcf84c9b7b27b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([11, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_966d63af3dd526c12e9e58f8852acee8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([4, 96, 96, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9848e8dff9606cffd537a985dec7d02f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a29428cd9939fed82f7390bef076a553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6998c4ea21d4dca0b465c039ca5ca40c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_971434b1d1259366713a05719b599893(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_767136b47697554ccb5d348991a12809(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 24, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dbfd795603daaec6d2015e68ec1f605(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13cc2e40d11de2b71c856b05a378f23f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([43, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e343a7e8882135b2bbfc20cbac7696ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1280, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6998c4ea21d4dca0b465c039ca5ca40c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([43, 768, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b23b5bb3245abea4d2790425d331bd69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1983b3e9534069e6bc82798d19f38015(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 52, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850c1eeae569985b821b627f27907744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_585343d893aa8f0aebdc4fe61887a5be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8fa5d53f4b60c330cf4c3f6d1e4a315(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01826525dae847acb4b1152cc0e0cac8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 20, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c770c4e1f6e4a3e29e3cdd2af80197b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe3c451f4ad03b8a06a96b13d8975f8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 92, 92], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_850c1eeae569985b821b627f27907744(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fe09d75d480fc8a91a916cd6030e998(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_985be251483665daa6648aad2019aec4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be1b80173f74cdf2d52d46ef25bc30bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 15, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56ce515b69d03914a42d3846514b8d80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c3d0dfa8611ec4fdb5902de31a39d73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 91, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da64ea60645fbae49f09359fbe144294(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 20, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_788d2b110b4d65e4b2a094aefa6561f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e54512e41bc31b8770f430ba4513c345(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_788d2b110b4d65e4b2a094aefa6561f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8cb2acaac2f70a6483d2f6150559d939(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([11, 384, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55007aae5da79e3ea3876865436c5ec9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9362ab17c35b3bd323ff37487d965fb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 38, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a3216bfd1a7cb61d41484d1a70e1ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f91c23913fb890ec120bcb90850e257(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_74203d898a1b9fa6328684937cd836e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 42, 42], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90d2137641bd40de96538b9fa6660aad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c90fd671566359d33f8b9faed7e89fda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8acf29de189f8f04300347483910fcfa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_42755af6b1f1376cdf696d8ee88f14dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f5d20df1fe654a144b600155268ebe5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20842cafd555c72d0cb73b2af2367f78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 5, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e295e2427381c81dccf6142d329f961c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da2ee72f9c7f9ec81888cf84cc537553(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2d476b7f78bcdedb9a2de04c962e2ebc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 68, 68, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_020334da83088182891e5a0e6a084e3a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb15081c00c457334dc0f5f82f250029(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 4, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2fb394cdc81ac047938c45a2f1a44df4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 48, 48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1144dbe4e8205fa1216e05db7bd06fcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37068a43619314911c77b9dc82a2362f
        def get_inputs(self):
            return [
                paddle.uniform([43, 704, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c69120bbcf741bcaefb4cf8f81a7c6e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1aafb48aae46c6464dce9df9c85c29cc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5a6985ed07a5a03baffc9c69c0ce9782
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()