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
    class PrimitiveOp_9f0118abd5e2332ebddf6bba1a881b45(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'bilinear', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71b2c32331a317e33951bbdddfd6eb67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f0118abd5e2332ebddf6bba1a881b45
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f57e5e553a313234845efe2088e047f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [4, 4], 'bilinear', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ab89e93175dfe67a9e15ddc0c943e3cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f57e5e553a313234845efe2088e047f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_60d72fdf4f81d9a0cd64e26ec61c265f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [8, 8], 'bilinear', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b2e34ebe61944ea6863700e5586f6d59(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60d72fdf4f81d9a0cd64e26ec61c265f
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9df99f7b1e2a990d7164fc7c9094392c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'bilinear', True, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 19, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f2d87f1f6bfe6e3b41e9bfe00f205157(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9df99f7b1e2a990d7164fc7c9094392c
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_88eeea3749c25d79ef74c5e11db30514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9df99f7b1e2a990d7164fc7c9094392c
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7960f1f9829bb477fef43b4803c9c32a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'bilinear', True, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_082cf190c8c7f6e089843570b75580a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7960f1f9829bb477fef43b4803c9c32a
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7156595dfa83217a1130857f77ac4a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7960f1f9829bb477fef43b4803c9c32a
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_632441922bfd16948546864af3030fa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7960f1f9829bb477fef43b4803c9c32a
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8972fc6ce2c3dbe18508adbefb0cc490(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 19, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf199c66564e7cc4b14290014dbe5a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8972fc6ce2c3dbe18508adbefb0cc490
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_a7f0ae3974180b2069fa19a663715379(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f0118abd5e2332ebddf6bba1a881b45
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0afe3130aef7199ddaa34c54c3dfd4e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f57e5e553a313234845efe2088e047f0
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0e5eee7a40600208542c0964c2ae1fbd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60d72fdf4f81d9a0cd64e26ec61c265f
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d2bee6d466dcfb07cac5b2ebf0eeb0a1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 150, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[2], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6fff9e1658b9268b3d287f095a44931a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2bee6d466dcfb07cac5b2ebf0eeb0a1
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 2], dtype='int32').reshape([2]),
            ]


    
    class PrimitiveOp_06aafd2c98371a7d873da7d057713afa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [2, 2], 'bilinear', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2db9b1bb1e5ea01c68d455209644518e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06aafd2c98371a7d873da7d057713afa
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 80, 120], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_43dee8d78ee13bb536a6994ba5ee0cd0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [4, 4], 'bilinear', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ebf3e3babb71f9cfab05320c32c4770e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43dee8d78ee13bb536a6994ba5ee0cd0
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 40, 60], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_989caa5834a884239a885edb09c8fc6b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, None, None, None, 'NCHW', -1, -1, -1, [8, 8], 'bilinear', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_45e9c942131bea50a431288e3faf469c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_989caa5834a884239a885edb09c8fc6b
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7156595dfa83217a1130857f77ac4a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7960f1f9829bb477fef43b4803c9c32a
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_632441922bfd16948546864af3030fa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7960f1f9829bb477fef43b4803c9c32a
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_082cf190c8c7f6e089843570b75580a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7960f1f9829bb477fef43b4803c9c32a
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7156595dfa83217a1130857f77ac4a6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7960f1f9829bb477fef43b4803c9c32a
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_632441922bfd16948546864af3030fa8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7960f1f9829bb477fef43b4803c9c32a
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 256, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_24d63b6a37e111c0aa3e758d7c83d1d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            input_2 = None
            input_3 = None
            return paddle._C_ops.bilinear_interp(input_0, input_1, None, None, 'NCHW', -1, -1, -1, [], 'bilinear', False, 0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e125596241451ff193212b9509aa2cdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24d63b6a37e111c0aa3e758d7c83d1d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_b4f6a689ce7d723c92d51df77aace624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06aafd2c98371a7d873da7d057713afa
        def get_inputs(self):
            return [
                paddle.uniform([1, 36, 88, 132], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de4b3b0c62288684c1c7be562f06541a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_43dee8d78ee13bb536a6994ba5ee0cd0
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 44, 66], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b4916b196024279216754e70163290c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_989caa5834a884239a885edb09c8fc6b
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 22, 33], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f4c1b0f2fe88b51a5bbd36034b6a599(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24d63b6a37e111c0aa3e758d7c83d1d6
        def get_inputs(self):
            return [
                paddle.uniform([1, 150, 60, 60], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([8, 2], dtype='int32').reshape([2]),
            ]


    

if __name__ == '__main__':
    unittest.main()