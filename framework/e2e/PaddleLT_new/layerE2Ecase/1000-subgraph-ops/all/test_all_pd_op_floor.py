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
    class PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbbda724a640896b305d4505d6dfba7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3271141052246094]]], [[[1.3329064846038818]]], [[[1.7705596685409546]]], [[[1.338937520980835]]], [[[1.326549768447876]]], [[[1.795101284980774]]], [[[1.3662755489349365]]], [[[1.1189559698104858]]], [[[0.9884886741638184]]], [[[1.461920976638794]]], [[[1.8237712383270264]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_c0b4a99af243a0b18e16058b33d769b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0b4a99af243a0b18e16058b33d769b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_568819835090acf7961f0252c66e84fa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb8bc341d8ac6d7d7f3dbbb1746ef959(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.111026644706726]]], [[[0.8947897553443909]]], [[[1.4013948440551758]]], [[[1.6212403774261475]]], [[[1.4724621772766113]]], [[[1.471761703491211]]], [[[1.2215818166732788]]], [[[1.8709466457366943]]], [[[1.8487491607666016]]], [[[1.8387360572814941]]], [[[1.797884225845337]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_814d8180c5701bb3421e8de21b509519(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3355836868286133]]], [[[1.4446052312850952]]], [[[1.6705386638641357]]], [[[1.312016248703003]]], [[[1.044058084487915]]], [[[1.3989274501800537]]], [[[1.5397329330444336]]], [[[1.2582091093063354]]], [[[1.2062907218933105]]], [[[1.569000005722046]]], [[[1.0314924716949463]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_c0b4a99af243a0b18e16058b33d769b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5f93c50c9e998705022e1cf13144e7c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17c0313ff681dbe82e4f2e15b7520a06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f93c50c9e998705022e1cf13144e7c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8241dae4f25ad67669e9378ffbdad579(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0b4a99af243a0b18e16058b33d769b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7133a0a8436cec455f2b466bc032e080(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_55b62b1ec7a43b4690fa241603f3ed24(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa78ec11f354254f5b9e26f1a0563e08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.972045660018921]]], [[[1.2322757244110107]]], [[[1.002550482749939]]], [[[1.147111177444458]]], [[[1.1341710090637207]]], [[[1.3422062397003174]]], [[[1.9006317853927612]]], [[[1.3410320281982422]]], [[[1.0543323755264282]]], [[[1.847813606262207]]], [[[1.6192326545715332]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_92044170c58faf69e110014fb78dff5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa6538fe55a1b08cbfe20d47155e4e5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f93c50c9e998705022e1cf13144e7c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6cc971fd9fbf39eef3287dc94a1cf317(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b17f3a3b45497ecd01d739fa1bf4a07b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6135d31a79910bbd6fcec14fefc4b7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f93c50c9e998705022e1cf13144e7c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_21e8679c4d52bd4bd1fa39d0a2c9a0a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_856081fa8935131929840486d5b99705(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86b0ac006b596632c383da460e07e5d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37df0be5f1f68f19659c1ab77b9b850d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f93c50c9e998705022e1cf13144e7c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45aeebb6c018aa79bea3a6a93c133e8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.641852617263794]]], [[[0.9537301659584045]]], [[[1.4906723499298096]]], [[[1.2271770238876343]]], [[[1.547351598739624]]], [[[1.5376088619232178]]], [[[1.5977516174316406]]], [[[1.457284688949585]]], [[[1.7084970474243164]]], [[[1.5453383922576904]]], [[[1.3102967739105225]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_c0b4a99af243a0b18e16058b33d769b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f254bfff0edb8c6d4525bfea38ed7391
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ee14f60bc5fb3c9219b346ff40cec5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36fae1aa2a56d654100cfbd298d10a26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5f93c50c9e998705022e1cf13144e7c0
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bdcb79f022c5a8dc55e906e77638f98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e4b2ea34a10891ff4a9da464e8375e
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d43061fbeededcfbdcffd15e3450471(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3271141052246094]]], [[[1.3329064846038818]]], [[[1.7705596685409546]]], [[[1.338937520980835]]], [[[1.326549768447876]]], [[[1.795101284980774]]], [[[1.3662755489349365]]], [[[1.1189559698104858]]], [[[0.9884886741638184]]], [[[1.461920976638794]]], [[[1.8237712383270264]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_e546468b613a789de087261ca576213f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e546468b613a789de087261ca576213f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.floor(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b03d9329b6847392d5293f53cf5d64c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_629c976faf72b50e61fe9ef71444dcc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.111026644706726]]], [[[0.8947897553443909]]], [[[1.4013948440551758]]], [[[1.6212403774261475]]], [[[1.4724621772766113]]], [[[1.471761703491211]]], [[[1.2215818166732788]]], [[[1.8709466457366943]]], [[[1.8487491607666016]]], [[[1.8387360572814941]]], [[[1.797884225845337]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_14957604514e4fa2099c8c2b7172c5d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.3355836868286133]]], [[[1.4446052312850952]]], [[[1.6705386638641357]]], [[[1.312016248703003]]], [[[1.044058084487915]]], [[[1.3989274501800537]]], [[[1.5397329330444336]]], [[[1.2582091093063354]]], [[[1.2062907218933105]]], [[[1.569000005722046]]], [[[1.0314924716949463]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_e546468b613a789de087261ca576213f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5fb96db48da856e70bf28496c270df07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c30f3f92a6a0d848951d29547a69c1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e546468b613a789de087261ca576213f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d92614b4f560ef0395b23157e3cf63a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_72ded714fbbfe488bbd7359633dabec5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2df0a98a6de46778d7f38d88c1389bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.972045660018921]]], [[[1.2322757244110107]]], [[[1.002550482749939]]], [[[1.147111177444458]]], [[[1.1341710090637207]]], [[[1.3422062397003174]]], [[[1.9006317853927612]]], [[[1.3410320281982422]]], [[[1.0543323755264282]]], [[[1.847813606262207]]], [[[1.6192326545715332]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_52b6eac51df9b08a666eae377dc7f9d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_48bfd158ddb00259f054b9f5a6692a00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_750c5665240151d11bc0672ce766d778(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_377e5d2cb8ca99ff6f2e1e10c8cb567c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17fbca6c3d322ce6a1d9e3ec15606ecd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b34d22d828d8a75a0f9f96e25d584a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d3b5f47e583442e5921b6f9ca537db8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_324f6f7d819b1d56754b6a66ce4c5a59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a7dd44253be8aa9923f544ff3283bd52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1aa47108fbd056f2d5fa1b7978884a83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[[1.641852617263794]]], [[[0.9537301659584045]]], [[[1.4906723499298096]]], [[[1.2271770238876343]]], [[[1.547351598739624]]], [[[1.5376088619232178]]], [[[1.5977516174316406]]], [[[1.457284688949585]]], [[[1.7084970474243164]]], [[[1.5453383922576904]]], [[[1.3102967739105225]]]], dtype='float32').reshape([11, 1, 1, 1]),
            ]


    class TestPrimitiveOp_e546468b613a789de087261ca576213f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([43, 1, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_341a93d3ae19a627042fe43bf9f1e4f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8ef3a52605ab6ce92ad2a895f372079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a8413c147683ea9b7e2509a2c7d8e01a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b3b0295cc59a64fc03ed3312b9741482(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daf66729c55b7ed09f415c0f91cbf276
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()