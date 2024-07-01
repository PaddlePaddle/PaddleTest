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
    class PrimitiveOp_975c0b133d543679010182499c6a647d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08124cdc29ced688f3e0c13d84be7ebd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0, -0.9130434989929199, -0.8260869383811951, -0.739130437374115, -0.6521739363670349, -0.5652173757553101, -0.47826087474823, -0.3913043439388275, -0.30434781312942505, -0.21739129722118378, -0.1304347813129425, -0.043478261679410934, 0.043478261679410934, 0.1304347813129425, 0.21739129722118378, 0.30434781312942505, 0.3913043439388275, 0.47826087474823, 0.5652173757553101, 0.6521739363670349, 0.739130437374115, 0.8260869383811951, 0.9130434989929199, 1.0], dtype='float32').reshape([24]),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_16c5f723c3683ad60c64eca814cb4c8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe8778ae29e6b8fb4050e276549fa6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16c5f723c3683ad60c64eca814cb4c8f
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dedb1b595b06a25f761383d7216f9fe9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[48], dtype='float32'),
                paddle.static.InputSpec(shape=[48], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd9535e19f9d729cfab5848504f9ea31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dedb1b595b06a25f761383d7216f9fe9
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d2a47810311d1f07cd2b6b26cee78afd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='float32'),
                paddle.static.InputSpec(shape=[24], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cd6fb9eb61d451f52cf7052fcc4d34f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2a47810311d1f07cd2b6b26cee78afd
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_fe8778ae29e6b8fb4050e276549fa6bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_16c5f723c3683ad60c64eca814cb4c8f
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd9535e19f9d729cfab5848504f9ea31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dedb1b595b06a25f761383d7216f9fe9
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c681b085f72a83aa551f14299832e80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2a47810311d1f07cd2b6b26cee78afd
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0], dtype='float32').reshape([24]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_a2356f8b776d9e5bc044614702f64381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
                paddle.uniform([38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c1a619bd16e388b7068b054616a06ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0, -0.8947368264198303, -0.7894737124443054, -0.6842105388641357, -0.5789473652839661, -0.4736842215061188, -0.3684210479259491, -0.2631579041481018, -0.15789473056793213, -0.05263157933950424, 0.05263157933950424, 0.15789473056793213, 0.2631579041481018, 0.3684210479259491, 0.4736842215061188, 0.5789473652839661, 0.6842105388641357, 0.7894737124443054, 0.8947368264198303, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([-1.0, -0.931034505367279, -0.8620689511299133, -0.7931034564971924, -0.7241379022598267, -0.6551724076271057, -0.5862069129943848, -0.517241358757019, -0.4482758641242981, -0.37931033968925476, -0.3103448152542114, -0.24137930572032928, -0.17241379618644714, -0.1034482792019844, -0.03448275849223137, 0.03448275849223137, 0.1034482792019844, 0.17241379618644714, 0.24137930572032928, 0.3103448152542114, 0.37931033968925476, 0.4482758641242981, 0.517241358757019, 0.5862069129943848, 0.6551724076271057, 0.7241379022598267, 0.7931034564971924, 0.8620689511299133, 0.931034505367279, 1.0], dtype='float32').reshape([30]),
            ]


    
    class PrimitiveOp_7dde03b2426e4578abf7e7ec81510827(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbf06ceab0e3ffac1332d40df554faff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dde03b2426e4578abf7e7ec81510827
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_af34c5aef5202c9793fcd1a3d9970c15(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_78c91513d49d77948a5e9a132b7dcb9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af34c5aef5202c9793fcd1a3d9970c15
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_474297f65d27c6aca972835e113bebea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='float32'),
                paddle.static.InputSpec(shape=[16], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc6a9d2d3d6a8518b4837fdea0ea61cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_474297f65d27c6aca972835e113bebea
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_dbf06ceab0e3ffac1332d40df554faff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dde03b2426e4578abf7e7ec81510827
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_78c91513d49d77948a5e9a132b7dcb9b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af34c5aef5202c9793fcd1a3d9970c15
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46a420fd5a9197e29d7494f52ed95e8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_474297f65d27c6aca972835e113bebea
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0], dtype='float32').reshape([16]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0], dtype='float32').reshape([16]),
            ]


    
    class PrimitiveOp_37f1b9b290ac3b1695c5149ff38ed330(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[80], dtype='float32'),
                paddle.static.InputSpec(shape=[80], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_abc0e16595e514a5cdc475bda4562555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37f1b9b290ac3b1695c5149ff38ed330
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a566340ffc531f9ec7b30377e0ba54d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='float32'),
                paddle.static.InputSpec(shape=[40], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09f825a1ff9f84535440a98032ba276c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a566340ffc531f9ec7b30377e0ba54d9
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f6d4c8c3d011c824a8c6b803945b52e7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='float32'),
                paddle.static.InputSpec(shape=[20], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_eedadee4b0f77fbada57f9a59c148e75(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6d4c8c3d011c824a8c6b803945b52e7
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_abc0e16595e514a5cdc475bda4562555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37f1b9b290ac3b1695c5149ff38ed330
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09f825a1ff9f84535440a98032ba276c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a566340ffc531f9ec7b30377e0ba54d9
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3866937f7aca0e3479ac93819dff43b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f6d4c8c3d011c824a8c6b803945b52e7
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0], dtype='float32').reshape([20]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ed096181a91005585cc6ebe8e442e45c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14], dtype='float32'),
                paddle.static.InputSpec(shape=[14], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08e41174795383a36a7555156bf09320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed096181a91005585cc6ebe8e442e45c
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0], dtype='float32').reshape([14]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0], dtype='float32').reshape([14]),
            ]


    
    class PrimitiveOp_0759def75905617422c610a89451b8d8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28], dtype='float32'),
                paddle.static.InputSpec(shape=[28], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_768eed7dbb476ba3f5ebe02e4d1b01ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0759def75905617422c610a89451b8d8
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0], dtype='float32').reshape([28]),
                paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0], dtype='float32').reshape([28]),
            ]


    
    class PrimitiveOp_89b445490b8065ac7a84fffbbad724e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56], dtype='float32'),
                paddle.static.InputSpec(shape=[56], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f7fa011f7e57180b184c79a3939c48ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89b445490b8065ac7a84fffbbad724e9
        def get_inputs(self):
            return [
                paddle.uniform([56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86a4ea45b43e48cf9723f6bde51146e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2a47810311d1f07cd2b6b26cee78afd
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0], dtype='float32').reshape([24]),
                paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0], dtype='float32').reshape([24]),
            ]


    
    class PrimitiveOp_80a453294ccd9e7c6c31506dc67212d3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[68], dtype='float32'),
                paddle.static.InputSpec(shape=[68], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbd26fe5e591a7289fa151967ee1914e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80a453294ccd9e7c6c31506dc67212d3
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_120e561a57eb0b2558ae57e959836da3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[34], dtype='float32'),
                paddle.static.InputSpec(shape=[34], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a27fc7df59c346b7bc67858c35e909ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_120e561a57eb0b2558ae57e959836da3
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_207aff601961081d9c1dc6e860bb510f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17], dtype='float32'),
                paddle.static.InputSpec(shape=[17], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cea07bcc940044a2655801f2da404089(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_207aff601961081d9c1dc6e860bb510f
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0], dtype='float32').reshape([17]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_cbd26fe5e591a7289fa151967ee1914e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_80a453294ccd9e7c6c31506dc67212d3
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a27fc7df59c346b7bc67858c35e909ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_120e561a57eb0b2558ae57e959836da3
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_17c6cee6259ec626a6f650f14a9c8fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_207aff601961081d9c1dc6e860bb510f
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0], dtype='float32').reshape([17]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41ac6e19817fd37e3c452d503c5aab7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0], dtype='float32').reshape([16]),
                paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_c615bbfcbd68b11254c9a767137026ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0], dtype='float32').reshape([8]),
                paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0], dtype='float32').reshape([8]),
            ]


    class TestPrimitiveOp_15de4e60f38ec9f7f591908df232999a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0, -0.8571428656578064, -0.7142857313156128, -0.5714285969734192, -0.4285714328289032, -0.2857142984867096, -0.1428571492433548, 5.551115123125783e-17, 0.1428571492433548, 0.2857142984867096, 0.4285714328289032, 0.5714285969734192, 0.7142857313156128, 0.8571428656578064, 1.0], dtype='float32').reshape([15]),
                paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
            ]


    
    class PrimitiveOp_3e6af9dcafd099128ba2af7605cfeb38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100], dtype='float32'),
                paddle.static.InputSpec(shape=[152], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cbcf421e51d119f4df1d2a92de862f68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3e6af9dcafd099128ba2af7605cfeb38
        def get_inputs(self):
            return [
                paddle.uniform([100], dtype='float32', min=0, max=0.5),
                paddle.uniform([152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_daaa9ec6ee844a01a9271ddff8813c89(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50], dtype='float32'),
                paddle.static.InputSpec(shape=[76], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_296cbcaa87a4c8af8d5ac14289325a7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_daaa9ec6ee844a01a9271ddff8813c89
        def get_inputs(self):
            return [
                paddle.uniform([50], dtype='float32', min=0, max=0.5),
                paddle.uniform([76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_933bfbd3b66d4d71e325b23cdbb1a938(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25], dtype='float32'),
                paddle.static.InputSpec(shape=[38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_662a6260455c4dc88f80f7d683f6c0b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_933bfbd3b66d4d71e325b23cdbb1a938
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0, 784.0], dtype='float32').reshape([25]),
                paddle.uniform([38], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1ed07715f148c77a8996215ca398c547(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13], dtype='float32'),
                paddle.static.InputSpec(shape=[19], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6379cd5d3ac4f01e697091745c01299(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ed07715f148c77a8996215ca398c547
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0], dtype='float32').reshape([13]),
                paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0, 1056.0, 1120.0, 1184.0], dtype='float32').reshape([19]),
            ]


    
    class PrimitiveOp_e95a63beea9a01b2dd8d10b0e0f9fa85(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7], dtype='float32'),
                paddle.static.InputSpec(shape=[10], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5db64d3604afef4651a6971bbff1cc2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e95a63beea9a01b2dd8d10b0e0f9fa85
        def get_inputs(self):
            return [
                paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0], dtype='float32').reshape([7]),
                paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0, 1088.0, 1216.0], dtype='float32').reshape([10]),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2bb376c1e910aacde0c117e2ab950d36(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[72], dtype='float32'),
                paddle.static.InputSpec(shape=[72], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_06dcdf190d886c154ee8f6c4e5c94a78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bb376c1e910aacde0c117e2ab950d36
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2f656780cef978b76da59374c29c82a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='float32'),
                paddle.static.InputSpec(shape=[36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ccead9d321eda3adf0d345e2c920784d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f656780cef978b76da59374c29c82a4
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_39b1f5bb624f8d268953849f3d49274d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0_0, arg_0_1):
            input_0 = [arg_0_0, arg_0_1]
            return paddle._C_ops.meshgrid(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[18], dtype='float32'),
                paddle.static.InputSpec(shape=[18], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_767d1fee0a51f17f06b63eec21641c18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39b1f5bb624f8d268953849f3d49274d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0], dtype='float32').reshape([18]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_06dcdf190d886c154ee8f6c4e5c94a78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2bb376c1e910aacde0c117e2ab950d36
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccead9d321eda3adf0d345e2c920784d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2f656780cef978b76da59374c29c82a4
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_604055163d9a185e156b3d5751c85a17(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39b1f5bb624f8d268953849f3d49274d
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0], dtype='float32').reshape([18]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_08124cdc29ced688f3e0c13d84be7ebd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0, -0.9130434989929199, -0.8260869383811951, -0.739130437374115, -0.6521739363670349, -0.5652173757553101, -0.47826087474823, -0.3913043439388275, -0.30434781312942505, -0.21739129722118378, -0.1304347813129425, -0.043478261679410934, 0.043478261679410934, 0.1304347813129425, 0.21739129722118378, 0.30434781312942505, 0.3913043439388275, 0.47826087474823, 0.5652173757553101, 0.6521739363670349, 0.739130437374115, 0.8260869383811951, 0.9130434989929199, 1.0], dtype='float32').reshape([24]),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6557982ed2b9ce07b5cdf503b07654b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b91d636491f8c65a360991a83e40ffab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be724fd58956402d4d4c36365708fde0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0], dtype='float32').reshape([24]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0, 640.0, 672.0, 704.0, 736.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_6557982ed2b9ce07b5cdf503b07654b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b91d636491f8c65a360991a83e40ffab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
                paddle.uniform([48], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8ece65fff8c13a43db95cea600ba0b6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0], dtype='float32').reshape([24]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_a2356f8b776d9e5bc044614702f64381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
                paddle.uniform([38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c1a619bd16e388b7068b054616a06ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0, -0.8947368264198303, -0.7894737124443054, -0.6842105388641357, -0.5789473652839661, -0.4736842215061188, -0.3684210479259491, -0.2631579041481018, -0.15789473056793213, -0.05263157933950424, 0.05263157933950424, 0.15789473056793213, 0.2631579041481018, 0.3684210479259491, 0.4736842215061188, 0.5789473652839661, 0.6842105388641357, 0.7894737124443054, 0.8947368264198303, 1.0], dtype='float32').reshape([20]),
                paddle.to_tensor([-1.0, -0.931034505367279, -0.8620689511299133, -0.7931034564971924, -0.7241379022598267, -0.6551724076271057, -0.5862069129943848, -0.517241358757019, -0.4482758641242981, -0.37931033968925476, -0.3103448152542114, -0.24137930572032928, -0.17241379618644714, -0.1034482792019844, -0.03448275849223137, 0.03448275849223137, 0.1034482792019844, 0.17241379618644714, 0.24137930572032928, 0.3103448152542114, 0.37931033968925476, 0.4482758641242981, 0.517241358757019, 0.5862069129943848, 0.6551724076271057, 0.7241379022598267, 0.7931034564971924, 0.8620689511299133, 0.931034505367279, 1.0], dtype='float32').reshape([30]),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e7cee05ba98803bff632aa6a31176915(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0], dtype='float32').reshape([16]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c3f3cea0d39b935ffbfcea7ced3dea9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0], dtype='float32').reshape([16]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_805668f2b7b6e14f83505187c409c7df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cff0cf81677ad149cc02df64bfca05e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b22dc56f336100f7909db7b28fa0f5e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0], dtype='float32').reshape([20]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0, 576.0, 608.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_805668f2b7b6e14f83505187c409c7df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
                paddle.uniform([80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cff0cf81677ad149cc02df64bfca05e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
                paddle.uniform([40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be3bb9ff6d6a432be952615bd96b6f52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0], dtype='float32').reshape([20]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_624e1e0f51b520e922f9c64ddc355b48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0], dtype='float32').reshape([14]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0], dtype='float32').reshape([14]),
            ]


    class TestPrimitiveOp_86d6af9306ecea066ff9c61f8801501f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0], dtype='float32').reshape([28]),
                paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0, 392.0, 408.0, 424.0, 440.0], dtype='float32').reshape([28]),
            ]


    class TestPrimitiveOp_b2bdb06e7478260090d090a2a1ba47f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([56], dtype='float32', min=0, max=0.5),
                paddle.uniform([56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80fd73d12762675385351e7c43ac7fc8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0], dtype='float32').reshape([24]),
                paddle.to_tensor([8.0, 24.0, 40.0, 56.0, 72.0, 88.0, 104.0, 120.0, 136.0, 152.0, 168.0, 184.0, 200.0, 216.0, 232.0, 248.0, 264.0, 280.0, 296.0, 312.0, 328.0, 344.0, 360.0, 376.0], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_9034c43c41287f5a45b04eaca366706b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6896d228ac903dc82fba0d9be97c0db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f7fa1c352fda0f96eae65b85440ea3e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0], dtype='float32').reshape([17]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_9034c43c41287f5a45b04eaca366706b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
                paddle.uniform([68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d6896d228ac903dc82fba0d9be97c0db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
                paddle.uniform([34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac4f3373607f323409450bb7c0dcd589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0], dtype='float32').reshape([17]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0], dtype='float32').reshape([17]),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_41ac6e19817fd37e3c452d503c5aab7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0], dtype='float32').reshape([16]),
                paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_c615bbfcbd68b11254c9a767137026ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0], dtype='float32').reshape([8]),
                paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0], dtype='float32').reshape([8]),
            ]


    class TestPrimitiveOp_15de4e60f38ec9f7f591908df232999a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([-1.0, -0.8571428656578064, -0.7142857313156128, -0.5714285969734192, -0.4285714328289032, -0.2857142984867096, -0.1428571492433548, 5.551115123125783e-17, 0.1428571492433548, 0.2857142984867096, 0.4285714328289032, 0.5714285969734192, 0.7142857313156128, 0.8571428656578064, 1.0], dtype='float32').reshape([15]),
                paddle.to_tensor([-1.0, -0.9166666865348816, -0.8333333134651184, -0.75, -0.6666666865348816, -0.5833333134651184, -0.5, -0.4166666567325592, -0.3333333432674408, -0.25, -0.1666666716337204, -0.0833333358168602, 5.551115123125783e-17, 0.0833333358168602, 0.1666666716337204, 0.25, 0.3333333432674408, 0.4166666567325592, 0.5, 0.5833333134651184, 0.6666666865348816, 0.75, 0.8333333134651184, 0.9166666865348816, 1.0], dtype='float32').reshape([25]),
            ]


    class TestPrimitiveOp_18ec1b0e141b6e1e88b9767261d23c33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([100], dtype='float32', min=0, max=0.5),
                paddle.uniform([152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_25b36015b136c8f1caab4a58ae03f1cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([50], dtype='float32', min=0, max=0.5),
                paddle.uniform([76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_205bb03023a9b6b9218e19817d41288f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0, 592.0, 624.0, 656.0, 688.0, 720.0, 752.0, 784.0], dtype='float32').reshape([25]),
                paddle.uniform([38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3118eb972e3197a0cfe245774745aea8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0], dtype='float32').reshape([13]),
                paddle.to_tensor([32.0, 96.0, 160.0, 224.0, 288.0, 352.0, 416.0, 480.0, 544.0, 608.0, 672.0, 736.0, 800.0, 864.0, 928.0, 992.0, 1056.0, 1120.0, 1184.0], dtype='float32').reshape([19]),
            ]


    class TestPrimitiveOp_ac21213fa1093e1e295f0ad35a772736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0], dtype='float32').reshape([7]),
                paddle.to_tensor([64.0, 192.0, 320.0, 448.0, 576.0, 704.0, 832.0, 960.0, 1088.0, 1216.0], dtype='float32').reshape([10]),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeb5f3ea5f84a1ff62d83c84fe27d870(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e66b435f02e71b7e0eb73f5009c43304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_689d4ba7d63456d17f590ce7446425cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0], dtype='float32').reshape([18]),
                paddle.to_tensor([0.0, 32.0, 64.0, 96.0, 128.0, 160.0, 192.0, 224.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480.0, 512.0, 544.0], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_eeb5f3ea5f84a1ff62d83c84fe27d870(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
                paddle.uniform([72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e66b435f02e71b7e0eb73f5009c43304(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d19d01b04200871c6f0c1c8b224c1200(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0], dtype='float32').reshape([18]),
                paddle.to_tensor([16.0, 48.0, 80.0, 112.0, 144.0, 176.0, 208.0, 240.0, 272.0, 304.0, 336.0, 368.0, 400.0, 432.0, 464.0, 496.0, 528.0, 560.0], dtype='float32').reshape([18]),
            ]


    class TestPrimitiveOp_05fdee6c30bb4b9949d6786e6d4975f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f4eef3d5d7c7eee0583b6d6b6d6ba1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f0d99ea67bbdfbe3157f083829183565(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_975c0b133d543679010182499c6a647d
        def get_inputs(self):
            return [
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()