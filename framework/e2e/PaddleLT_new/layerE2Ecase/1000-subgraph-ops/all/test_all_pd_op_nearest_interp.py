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
    class PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'nearest', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1acf94b102234cfdea898897ff2a4300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc6728e08823cd4c2396f88433f2095b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157ac18eae5726aa30c352c86fdd2589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30ee35c053851797382a8fe414e2fc05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eeefea7c71fe68024b6b9d951d4d3da4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_718b3f1a79b7f0c48beb7b999ca90edd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9a142c393cbe3a23947e10fe4fc1c622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_718b3f1a79b7f0c48beb7b999ca90edd
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14a476a3b7250eef712e3510e0b56dad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_718b3f1a79b7f0c48beb7b999ca90edd
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df2621c297a24740d4b17eefa13a3217(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_718b3f1a79b7f0c48beb7b999ca90edd
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5fc701482a1e1d693acddb1b44e6b6bd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [8, 8], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_faa8d0b0ae6972eb7146fc85bbda362e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fc701482a1e1d693acddb1b44e6b6bd
        def get_inputs(self):
            return [
                paddle.uniform([2, 64, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0e0bc0b319bba3a3e3cc32f8520b6a43(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [4, 4], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cce3332f48135cb663593435e09675d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e0bc0b319bba3a3e3cc32f8520b6a43
        def get_inputs(self):
            return [
                paddle.uniform([2, 64, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2e0bb4a4ad018f029e922fe71cd0a226(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2c1406925d5b2919860eff7eb094bcb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e0bb4a4ad018f029e922fe71cd0a226
        def get_inputs(self):
            return [
                paddle.uniform([2, 64, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c6b73710f31ed56a68eeaf0421c974a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f11e795a20f7ace992139598161711b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_87659a9fff2f0fb90791291d9fb2e1e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d48fcece9b2098f18eed49fa34c19b46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e587b60670dee547afdfe64afd1ca555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dbd00fabf0d84e17e5fc9c8f6dcf591(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c463e2f67f2e5c0f4186ae8d6701306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a520fe8ea604e0aafeeec8e938f0ea44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_718b3f1a79b7f0c48beb7b999ca90edd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e9238c35b6a686f06732eca40e44b23(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_718b3f1a79b7f0c48beb7b999ca90edd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3bf0179cf253754bce8263b3919af4f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_718b3f1a79b7f0c48beb7b999ca90edd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01dba128b7a5d604ae1eb17360751c9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fc701482a1e1d693acddb1b44e6b6bd
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1727e5ce23a5f607075f9491e3a93ddc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e0bc0b319bba3a3e3cc32f8520b6a43
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c470f01fe50b5a591ee6edf0b445b0d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2e0bb4a4ad018f029e922fe71cd0a226
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ef7d15db4c90ce0268ef799a6d0f47a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80265550fa69e52a1b7ce11063247336(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04ad78d41a625ff1acd2040f379b2351(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f4b0f562fe38ca99a4e04176e8665a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0710f51043354222be693fe08fdfed47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a45d7c0c5f8d1c6f88a1938edbfc3b6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3552c79887ee1a1d0082f382f1ed379b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe0246ee7ae4be680519b2eb7aa4597c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1acf94b102234cfdea898897ff2a4300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc6728e08823cd4c2396f88433f2095b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5552762878845756831cdca80b7dd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8d055a0f8de652e95f5dcc6adbc188ba(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'nearest', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f10bdfe34d2744d3681744063f99ea7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d055a0f8de652e95f5dcc6adbc188ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6bccec62082f3c5b17b171879138ddf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d055a0f8de652e95f5dcc6adbc188ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6915fa632e56fe6a573a867f95f8d02e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8d055a0f8de652e95f5dcc6adbc188ba
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_157ac18eae5726aa30c352c86fdd2589(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30ee35c053851797382a8fe414e2fc05(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1acf94b102234cfdea898897ff2a4300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc6728e08823cd4c2396f88433f2095b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5552762878845756831cdca80b7dd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1acf94b102234cfdea898897ff2a4300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc6728e08823cd4c2396f88433f2095b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_648efdafec30a5f4a2b05b755e09254f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_159bcc6292cff807c0ca287e8c490c6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_648efdafec30a5f4a2b05b755e09254f
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_300ac8aefe0e237a90c3efb547c2db20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_648efdafec30a5f4a2b05b755e09254f
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e185a1798d05fa61faf2ac5a31dd4ee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_648efdafec30a5f4a2b05b755e09254f
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dababbbfa7705912533be7e895b81b62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [8, 8], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_762dde73a6cfc58451dcdab6082bfffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dababbbfa7705912533be7e895b81b62
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_49b8a4b2d9240ae3384b83f9bdce999a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [4, 4], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f1cc9645ab039439d3efeb0aecf8fe45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b8a4b2d9240ae3384b83f9bdce999a
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4bdeb82d0f74255d9ac0f63e8d2dd2f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 24, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b5f7e770f2629c4c4c3647709fd904bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bdeb82d0f74255d9ac0f63e8d2dd2f0
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f175f7c0ae100988402f1ce624e6022(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2dedec638311af835351ad1882231fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9090c847ca618495bd4beb5627f48ca6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f4b0f562fe38ca99a4e04176e8665a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0710f51043354222be693fe08fdfed47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ef7d15db4c90ce0268ef799a6d0f47a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80265550fa69e52a1b7ce11063247336(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1acf94b102234cfdea898897ff2a4300(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bc6728e08823cd4c2396f88433f2095b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5552762878845756831cdca80b7dd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dbd00fabf0d84e17e5fc9c8f6dcf591(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c463e2f67f2e5c0f4186ae8d6701306(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c69143e551931dfb70f02a3e4faf3e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa780d856489cadddedc6f6f686cf2a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62bff35841def96e977a34cd06f04a90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63f7fb75639c6763e2d5f712bb27c2bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d48fcece9b2098f18eed49fa34c19b46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e587b60670dee547afdfe64afd1ca555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d13c928ddd6146b900d1e5b97c4eadd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_00107e832affb7f6ab111b16a2f37093(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f61263017ab625b0e60b950bf2bf2e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_848da6ffd25094c35ed5692ef23f50a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be17134d3c171a1b4e1e1470aafe6117(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8eaee5d85dc20c81b2acd6fc847993e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cbdc25bc851fb61512a74c587c909c42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89a5cf1c9ff886f38287d9d8a3fc80d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_426b7cb631a64bba0b72b992ad096e29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2077a2dfeb235f7b453b69b5405bbf54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91f556ebfb342906d0d9a832d75eb053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_632244c4a645430339ae1cfe5f095f08(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_292d7a97189580543181facfb3febc0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_287b26fb109541d82edb9e3c09a3dd98
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29020de081dbaa45b7383c8e9111cf44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_648efdafec30a5f4a2b05b755e09254f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_055751b401995d1d8a88f364881d1910(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_648efdafec30a5f4a2b05b755e09254f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8149cbd5fc76179fe0262c866a57a38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_648efdafec30a5f4a2b05b755e09254f
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fec022a2de0f503658b0a41784499190(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dababbbfa7705912533be7e895b81b62
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8e5449f08913ebd0a0786766484628a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49b8a4b2d9240ae3384b83f9bdce999a
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c5a34451498d212bb1116c72cd896ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bdeb82d0f74255d9ac0f63e8d2dd2f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_000176c95f30fd4273671205a5a75079(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'nearest', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8342c0fccbe203cc1722cd96df0235d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9b49712fe6329d795791506fee5dc80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd9767f267f57278edc150b43ea54414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86342f8384a448c4afb1e1aec3207852(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0d590363646c3354b012697e22d2b8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 140], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d1435a731902dfc62956c3724cfc5f84(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f776894738c654af0ae911168c47fb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3a7303bc025e4a42b29afeab5feeb09b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebff88e14fc66603b13872b8655f09eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([2, 256, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d235f7869a18ce5a66ccfb6f5b8676ed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [8, 8], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_515b5dc577b45512f5645892db8c06ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d235f7869a18ce5a66ccfb6f5b8676ed
        def get_inputs(self):
            return [
                paddle.uniform([2, 64, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d95a657c8dee447f728afefbbe4f7204(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.nearest_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [4, 4], 'nearest', False, 1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e0784bbd486884dce476f5eac38124a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d95a657c8dee447f728afefbbe4f7204
        def get_inputs(self):
            return [
                paddle.uniform([2, 64, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_766feaf90dbc19c050fa2ae0ff954df1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([2, 64, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_94b99283ee97cc2903156d6533e709a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 22], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f20e5eafada069df88ce5e710ab862d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 44], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_610454116feb1e10a1fd958893e23446(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 88], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1289815c4fb63c5a482d297ab8f9d3d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5e000a615089ca12f496f744576bb40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed46c4d497308ef4f4db86ff8c1f605b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_647af49be2e57145fda45739d8eb20d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_696dfa4946a23ca779ff881867ddd12d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e3569f13e5713476cc911693b399ab8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6772ae7e7e26a20e60d00f6579469de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29602ec98ac85afdd3589fd911ccee2b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d235f7869a18ce5a66ccfb6f5b8676ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce8e06a0038b62fa72ba00fd8a955e48(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d95a657c8dee447f728afefbbe4f7204
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d275260b8d91a0dc75760e67d1811e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8099f781df53d7718a3957fec68ce8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_751a9dea15635d35798b89d4cef859ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6e61c8df216979d8b1fa591f7f73486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b099019e8404a320a8b4074ae75596c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_212276c8d48dd98e64090896ab5274e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7c32f6cbf1c28a1c0091e52f529e0f1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f69b4f7ec92c9a7588da3cbb3889d6cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 30, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80703a1302aadcf155cc98c98aa373c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 60, 100], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8342c0fccbe203cc1722cd96df0235d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9b49712fe6329d795791506fee5dc80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fc9f0436e8a8bf87e99353cff22c3d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12c0e7540f64ea4f0cd6f07682aa8763(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 17, 26], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92cbfc21b144f6af9604c90c92dcbba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 34, 52], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d6c1ceb59c76c75d3edcf2ac69f7c2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 68, 104], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd9767f267f57278edc150b43ea54414(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 23, 35], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86342f8384a448c4afb1e1aec3207852(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 46, 70], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8342c0fccbe203cc1722cd96df0235d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9b49712fe6329d795791506fee5dc80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fc9f0436e8a8bf87e99353cff22c3d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8342c0fccbe203cc1722cd96df0235d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9b49712fe6329d795791506fee5dc80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_32b1682383dbfb3f2434d19a46d92a19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_161e0b807fb0c2e0417bddfd33184806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_569e3ed6d48e4c0ccdcae707a53ee1f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([2, 96, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c139346d8247bb5e4bc1c8b7340cbbae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d235f7869a18ce5a66ccfb6f5b8676ed
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 30, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3c0f7417b0b684705bd9a19acda0928a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d95a657c8dee447f728afefbbe4f7204
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 60, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb4855df9da4b6c56b44cc76e9ad78e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([2, 24, 120, 120], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b94c307b5c6136e6c1d15ad3c8ca3433(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 17, 20], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d8dba897ea10bcdf426b4f63322d674e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 34, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b6b84b583371c30309670e87892008f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 68, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b099019e8404a320a8b4074ae75596c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_212276c8d48dd98e64090896ab5274e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8099f781df53d7718a3957fec68ce8db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_751a9dea15635d35798b89d4cef859ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8342c0fccbe203cc1722cd96df0235d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9b49712fe6329d795791506fee5dc80(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1fc9f0436e8a8bf87e99353cff22c3d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 152], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed46c4d497308ef4f4db86ff8c1f605b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_647af49be2e57145fda45739d8eb20d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 48, 72], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a721feacf14d083c64bb2cad850762b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 96, 144], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2549013f87bc5718dcf987bbf4541582(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 18, 27], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ceabde99752b0fcb8104f54337f1affd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 36, 54], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c12c004b2c4b2d6a7407df6b989ba50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 72, 108], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1289815c4fb63c5a482d297ab8f9d3d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e5e000a615089ca12f496f744576bb40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce16a9c6cef54c8d767ee6b2d6a595be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f93f58e774f6c26c5a8dbb73e915f0b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 21, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d432d8e463c1edc3fdaca85e77379d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 42, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdd02415553325627e3a32f7dbf2df9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 84, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6b85fe01cfc5c16da11efe317f55f797(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 34], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b0564e4254fcc58aacaba824554889a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 50, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3593d2fdf53996be78fb43616e19563(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 100, 136], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_219b7244ae6a7e6a3272f8ad5261c6f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 28, 40], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c557f1e32ebcdcbb1e7ed8317ee33e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 56, 80], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c863bf9d54717cc9ba844722433a8b00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1e8fb96f268823be35cc7df53228965a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 40, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eb36935d915142ee1ecd5744e8455b64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 19, 29], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9daaa2dbf027480e1555bcb72e27e0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_000176c95f30fd4273671205a5a75079
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 38, 58], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fba4fb4655bc1aab18a8e4bd0b6f5a70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43bd5bac868719df45446fd88ac10852(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a904e9f5588725b635f991cd179c5698(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbf3467a63345752a68e5b741823dcaf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d235f7869a18ce5a66ccfb6f5b8676ed
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 23, 41], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4fc4fdbf64ad2e6d2d6cfc07b125109f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d95a657c8dee447f728afefbbe4f7204
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 46, 82], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eab48482bb7a55db75c0b0b8fd95e513(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d1435a731902dfc62956c3724cfc5f84
        def get_inputs(self):
            return [
                paddle.uniform([1, 24, 92, 164], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()