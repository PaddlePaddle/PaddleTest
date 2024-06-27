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
    class PrimitiveOp_cdc7b4d17071c467a0d1e332a6348d00(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e8ee9f97a48940598814a14337b75cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdc7b4d17071c467a0d1e332a6348d00
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_62ab0a1e259e6c146e124c2052c322f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e710e32d36aed2974f2d861d1209d4cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62ab0a1e259e6c146e124c2052c322f2
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c6785aeb85a316dd5bf666fed2eb9712(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_408190f8f545aa77923f55a4c00e555b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6785aeb85a316dd5bf666fed2eb9712
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b2f097e060e3056648efa4523955b7a9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_615d5089923064b76a250151c39515ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2f097e060e3056648efa4523955b7a9
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_18b92ebe5c2c2168533c86b1a8db4c20(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf20ed8cc59816fdff31853fc36bb034(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18b92ebe5c2c2168533c86b1a8db4c20
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85a2b6dcbce77e8aafd23162c526d2d0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1c1eee5b2bde9ff86c4875ebbfd7aa5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85a2b6dcbce77e8aafd23162c526d2d0
        def get_inputs(self):
            return [
                paddle.uniform([16, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7edc30136aa5d1f66aeb085fbdd31b02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8090323cbe08529fc57477e2a9827153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7edc30136aa5d1f66aeb085fbdd31b02
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0df2cc75ccd6a284b012797234fdc69(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ded5252bf2f246e722d02fe337bf5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0df2cc75ccd6a284b012797234fdc69
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_51c206f40e563f9f334bad87d60bcfb3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98a89319a334c94e409c55547af9aad6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51c206f40e563f9f334bad87d60bcfb3
        def get_inputs(self):
            return [
                paddle.uniform([128, 32, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5189409ef4f37a81e5f52413053c13b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e3010953691d81c0677bff57e41b44d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_06f9129d3cf9b7154815e3b5917a5678(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0d28b91bec0a6878851c819881ae46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06f9129d3cf9b7154815e3b5917a5678
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d683dc59a7919c0fefb49ba081eaf320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62ab0a1e259e6c146e124c2052c322f2
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_457692c642c942c0313d56884d84bc1d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9616b1757c32e3bf9820740c40a8be1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_457692c642c942c0313d56884d84bc1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a96c44547242a1f680b94ad8219aff44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51c206f40e563f9f334bad87d60bcfb3
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fcb5ebc91a181687d19cfb7c421a310d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8192, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe46e529872c86b28b39c9d6d27356cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcb5ebc91a181687d19cfb7c421a310d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_719232b86fd3ac719a6b0a1eb8d4a8ad(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89ab7a791f7993a6f6e7b12fab71ce00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_719232b86fd3ac719a6b0a1eb8d4a8ad
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fab4615f9e58c13a0a68827992307c91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_98801eadf1f6c075ca0b8e39b804793d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fab4615f9e58c13a0a68827992307c91
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bbd696d453f8c3344c9e2f932b538cc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9a18a06484008bcccbd06eec465bcc0c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7f40a3ea3a11002a44b59a399756bfff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a18a06484008bcccbd06eec465bcc0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_01b547b0e332e927778ddbd0c7073072(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8848888797db84ce666bb757a1403a44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b547b0e332e927778ddbd0c7073072
        def get_inputs(self):
            return [
                paddle.uniform([64, 256, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a1227a7bbd7ccfd8c646a7ad850a13ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c58435856777a04a4df2163ba36da94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1227a7bbd7ccfd8c646a7ad850a13ee
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ded5252bf2f246e722d02fe337bf5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0df2cc75ccd6a284b012797234fdc69
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0661232bcecf8b34e5914c35fdc21aef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5358481b40b7338516de2263ab8256e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0661232bcecf8b34e5914c35fdc21aef
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0d28b91bec0a6878851c819881ae46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06f9129d3cf9b7154815e3b5917a5678
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4c8c4888433f8726b788878e68e7f778(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04e9020230a517b525674f3227d31087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c8c4888433f8726b788878e68e7f778
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b598158a266511e5e1d7c12012df68a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85a2b6dcbce77e8aafd23162c526d2d0
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_01268f0615b712366268da427dbc5477(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_09be94b0be69a719314df6ae99f0e128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d7b447b78dd7c3749bf5fba3687e5af8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 16384, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4aab1fb3a5cc05dcd99de3963c0c36f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b447b78dd7c3749bf5fba3687e5af8
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a11f046d7a986659774eb0ceb9f98603(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_719232b86fd3ac719a6b0a1eb8d4a8ad
        def get_inputs(self):
            return [
                paddle.uniform([8, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e3010953691d81c0677bff57e41b44d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7901c4f1adddce863f501a1616dbc172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_34885c63dcad6c7be6ff7246afbd7781(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c971c5c4f9e78454bc881de6554ca570(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34885c63dcad6c7be6ff7246afbd7781
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_59b3be79653890b8ca282a54d809428b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6785aeb85a316dd5bf666fed2eb9712
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_657b7f505cd63962b31f18d8d9699016(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f75f8432af58ceee74524ebd4b1cfd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_657b7f505cd63962b31f18d8d9699016
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1e2a64719a57d6dd93b73f4cd3fdee8b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a9229e7f79ef4343ee33a87bcbe2c14c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2a64719a57d6dd93b73f4cd3fdee8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9229e7f79ef4343ee33a87bcbe2c14c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1e2a64719a57d6dd93b73f4cd3fdee8b
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef7f48616fb5cd88d0b09063747b7174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_62401122611c8b65d40ccec2f0e3b084(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_30745a59f98c5609f6ed8c361afbfd6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62401122611c8b65d40ccec2f0e3b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_036dba484b2ba03d0957f2b88387e71d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0e21368715842b19434cf0f8a6957035(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_08238ae158716b3939ba0bf1fb35a168(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e21368715842b19434cf0f8a6957035
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c4665d4980b92f5773b6f0b5a401b5ef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_364fe1760d0e9ed25fb1dff166bcea6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4665d4980b92f5773b6f0b5a401b5ef
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e710e32d36aed2974f2d861d1209d4cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62ab0a1e259e6c146e124c2052c322f2
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_554f030faec0df09b08a63813801d71e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09be94b0be69a719314df6ae99f0e128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bbd8b5b6cd97a23527145213a73ee8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1227a7bbd7ccfd8c646a7ad850a13ee
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bbd8b5b6cd97a23527145213a73ee8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1227a7bbd7ccfd8c646a7ad850a13ee
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20e206ba3fdc5c8368359a4b41396eca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09be94b0be69a719314df6ae99f0e128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3fc3f7eda511abe9ea49afb222db24e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbd696d453f8c3344c9e2f932b538cc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e576b96817995ac8dffd34bc1376c95e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_07f64a063497991f7248b35f4187033c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e576b96817995ac8dffd34bc1376c95e
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e28a747916be96985499781ff7fc2c0d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9e86b635dbc51271fb5d825b7c3178c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28a747916be96985499781ff7fc2c0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_36df1b792b87d42c8a9c9935461e131d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e2187e9a3369c562d5bf97d8a1a8e5c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36df1b792b87d42c8a9c9935461e131d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69491868bcae946ec3176fbd300b3ecb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e21368715842b19434cf0f8a6957035
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0e3f47948f259d9de1a32baa466d8d87(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_65daf24669abc61df512f9b9ffa31b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3f47948f259d9de1a32baa466d8d87
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe5d82b398bf5f21a9b8fc85f55c7f18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b547b0e332e927778ddbd0c7073072
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ec6d8064e09fa7da6d82219a0a2562d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 100, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_393e8f16e2ed4b23e2d64553a88e3c2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec6d8064e09fa7da6d82219a0a2562d1
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8ee9f97a48940598814a14337b75cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdc7b4d17071c467a0d1e332a6348d00
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ee65c015d91875934e5f9a17f1dc21af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6785aeb85a316dd5bf666fed2eb9712
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9b97be7f85638dc58c790c9c9fee0be(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62401122611c8b65d40ccec2f0e3b084
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f40a3ea3a11002a44b59a399756bfff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a18a06484008bcccbd06eec465bcc0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4702bbfac419859312442db9e334e3d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14e431ee0370071121789f4ed1091c4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4702bbfac419859312442db9e334e3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a20007f98d3f8bd7f23ef84b6a59e87(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f0ca241a7e9b4dd50e735636553aca6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a20007f98d3f8bd7f23ef84b6a59e87
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_205d523749c6d9f5dd3888c0a7214c70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1227a7bbd7ccfd8c646a7ad850a13ee
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef7f48616fb5cd88d0b09063747b7174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_52ad6ec3904cc074ee046fc8a3ea60ea(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f0aa531bcfd7dfc46bab0631a2647a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52ad6ec3904cc074ee046fc8a3ea60ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_55b0b1b45f42e1f96a285df781d3fcf2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a29468e2b79ce7ad8ef80d110a0c706c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55b0b1b45f42e1f96a285df781d3fcf2
        def get_inputs(self):
            return [
                paddle.uniform([8, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84f57563a0859d05de5f5af1491d2b42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f75f8432af58ceee74524ebd4b1cfd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_657b7f505cd63962b31f18d8d9699016
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2300fcbb0409d8aa5aa451b1f3a775f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d683dc59a7919c0fefb49ba081eaf320(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_62ab0a1e259e6c146e124c2052c322f2
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04e9020230a517b525674f3227d31087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c8c4888433f8726b788878e68e7f778
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d8acfc9b4519ed8f80175d6d3f7a484c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f68fb14c8ba77ceaa742cb7c91f5798(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8acfc9b4519ed8f80175d6d3f7a484c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8b0e919c9b5c9c1e220b40aaff8887b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34885c63dcad6c7be6ff7246afbd7781
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5bb2e6cfe5caf301fcd191a265068dc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef7f48616fb5cd88d0b09063747b7174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0343a32409ef3e0794ee6c1ad4defa53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 50, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_917f31df658144085f22ca0f76ecf648(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0343a32409ef3e0794ee6c1ad4defa53
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0bee6f9bf9037750d1a54b2006061a18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8bbd8b5b6cd97a23527145213a73ee8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1227a7bbd7ccfd8c646a7ad850a13ee
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9616b1757c32e3bf9820740c40a8be1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_457692c642c942c0313d56884d84bc1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c14fbfb9016cc514584add85ee454ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0661232bcecf8b34e5914c35fdc21aef
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_549efa21d993482520c268e8e8a35987(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2048, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e826f2679dc8c0eaf5f2ec7905f5a2a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549efa21d993482520c268e8e8a35987
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_951c4f8c35881f3e9b20f596a3be257a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32c3b6a8f86103659d95033d168129d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_951c4f8c35881f3e9b20f596a3be257a
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c42759c8b4fe8a463c98c83bae370732(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84f57563a0859d05de5f5af1491d2b42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbd696d453f8c3344c9e2f932b538cc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c42759c8b4fe8a463c98c83bae370732(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30f7eb322efcf03bbc70cc05c77b1512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e3010953691d81c0677bff57e41b44d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2300fcbb0409d8aa5aa451b1f3a775f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbd696d453f8c3344c9e2f932b538cc4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c7a99b6f561a0e476ec6e8cd008fba8d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2abeadabb3cf88a63fd596c03d373c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7a99b6f561a0e476ec6e8cd008fba8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e3010953691d81c0677bff57e41b44d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_036dba484b2ba03d0957f2b88387e71d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ef7f48616fb5cd88d0b09063747b7174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ae023a3d25c30e6782582b8a0b634724(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4096, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aba87c5d9c47243014698e1e78459d65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae023a3d25c30e6782582b8a0b634724
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9ff5c340b0a2be537ee75777c23562bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_951c4f8c35881f3e9b20f596a3be257a
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f0aa531bcfd7dfc46bab0631a2647a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52ad6ec3904cc074ee046fc8a3ea60ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_61c2b86b8e8bc856b2d0820024c9f105(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36df1b792b87d42c8a9c9935461e131d
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8090323cbe08529fc57477e2a9827153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7edc30136aa5d1f66aeb085fbdd31b02
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bedf94b84bdf4c4e52b382b8d8dcd28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7a99b6f561a0e476ec6e8cd008fba8d
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c58435856777a04a4df2163ba36da94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1227a7bbd7ccfd8c646a7ad850a13ee
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1af2c3301328373e3c60b1c1ce74a2d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1227a7bbd7ccfd8c646a7ad850a13ee
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_615d5089923064b76a250151c39515ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2f097e060e3056648efa4523955b7a9
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84f57563a0859d05de5f5af1491d2b42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_364fe1760d0e9ed25fb1dff166bcea6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4665d4980b92f5773b6f0b5a401b5ef
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c58435856777a04a4df2163ba36da94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a1227a7bbd7ccfd8c646a7ad850a13ee
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_09be94b0be69a719314df6ae99f0e128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c79d91559c6f1490e9e38b148be12efe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4aab1fb3a5cc05dcd99de3963c0c36f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b447b78dd7c3749bf5fba3687e5af8
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a35aa2e8fffb899adfecc82dae22bc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8acfc9b4519ed8f80175d6d3f7a484c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe46e529872c86b28b39c9d6d27356cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcb5ebc91a181687d19cfb7c421a310d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f68fb14c8ba77ceaa742cb7c91f5798(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8acfc9b4519ed8f80175d6d3f7a484c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aba87c5d9c47243014698e1e78459d65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae023a3d25c30e6782582b8a0b634724
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_024035e36e5cb37f2882f2fdd4f8c3bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d140eff45fc36bd74976174f0ec17ebe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_024035e36e5cb37f2882f2fdd4f8c3bc
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b2eddde1a2a503b308f096a143469686(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6785aeb85a316dd5bf666fed2eb9712
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65daf24669abc61df512f9b9ffa31b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3f47948f259d9de1a32baa466d8d87
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1f19cb524770cab4ac9d871a0ca666b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4702bbfac419859312442db9e334e3d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e826f2679dc8c0eaf5f2ec7905f5a2a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549efa21d993482520c268e8e8a35987
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ea8784ba758bbc430dc90e153d8de3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_024035e36e5cb37f2882f2fdd4f8c3bc
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84f57563a0859d05de5f5af1491d2b42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3519a81a5784874414b46a3183c09ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e21368715842b19434cf0f8a6957035
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3f3b2fc232fe64c02e590a549541f024(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4ae5cba55aa2b16f599ec031849f3393
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9e86b635dbc51271fb5d825b7c3178c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28a747916be96985499781ff7fc2c0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eacfeaca6faeddaf63b5f33776386d14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55b0b1b45f42e1f96a285df781d3fcf2
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2ca9da0f87ae1d378077593bb0914e77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5189409ef4f37a81e5f52413053c13b5
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf20ed8cc59816fdff31853fc36bb034(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18b92ebe5c2c2168533c86b1a8db4c20
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a35aa2e8fffb899adfecc82dae22bc9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d8acfc9b4519ed8f80175d6d3f7a484c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2300fcbb0409d8aa5aa451b1f3a775f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c2300fcbb0409d8aa5aa451b1f3a775f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01268f0615b712366268da427dbc5477
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a6993af301ec8385e211e6fcd4c3ce9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e21368715842b19434cf0f8a6957035
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8ee9f97a48940598814a14337b75cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdc7b4d17071c467a0d1e332a6348d00
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fe0f0267aef87aa4f5f66e1c7dcee096(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_762906817052d989a8b48a47938e0ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe0f0267aef87aa4f5f66e1c7dcee096
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_89230db5357ae99602541c02fbf7b604(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 198, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ed1f41ac40a33af8a387da356c68d93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89230db5357ae99602541c02fbf7b604
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_615d5089923064b76a250151c39515ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2f097e060e3056648efa4523955b7a9
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf20ed8cc59816fdff31853fc36bb034(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18b92ebe5c2c2168533c86b1a8db4c20
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_68635f3c46e201109c51ea90bba670b3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 1024, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_76681fc230d8d9026d8ac13f28240d03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68635f3c46e201109c51ea90bba670b3
        def get_inputs(self):
            return [
                paddle.uniform([16, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8090323cbe08529fc57477e2a9827153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7edc30136aa5d1f66aeb085fbdd31b02
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ded5252bf2f246e722d02fe337bf5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0df2cc75ccd6a284b012797234fdc69
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f265311f72cd45d894ba3f8f9cff2556(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[128, 32, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_924654d0095e4a87094ddc1b66161b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f265311f72cd45d894ba3f8f9cff2556
        def get_inputs(self):
            return [
                paddle.uniform([128, 32, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f79df13980bb3ddf78cad467c8eef448(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 49, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9d7b1a49528649bc175c5bd69a42e9fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f79df13980bb3ddf78cad467c8eef448
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0d28b91bec0a6878851c819881ae46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06f9129d3cf9b7154815e3b5917a5678
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9cd1d794992abe22bd83eceb48021dcd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9cbe492a8ca448e5f4e72deeaba6776a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cd1d794992abe22bd83eceb48021dcd
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9616b1757c32e3bf9820740c40a8be1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_457692c642c942c0313d56884d84bc1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_97ee1d166ffbf92ba896a4b1ef13982c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 256, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_882427e7549b41dfcfad2454d9a9b8b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97ee1d166ffbf92ba896a4b1ef13982c
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe46e529872c86b28b39c9d6d27356cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcb5ebc91a181687d19cfb7c421a310d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_54879a6d25143cfa970f75ec10fd7f9b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 256, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5763defb10edff0a51eb99ced9946a6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_54879a6d25143cfa970f75ec10fd7f9b
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e485346185c9f54ad0a5fe6eac79266d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 577, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e9f3ab4313725456e352cd020b151e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e485346185c9f54ad0a5fe6eac79266d
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3405df2db121487fbd00faf3656a32f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 196, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44f55890cb924acafa3504feef6a42ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3405df2db121487fbd00faf3656a32f
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f40a3ea3a11002a44b59a399756bfff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a18a06484008bcccbd06eec465bcc0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b3d2cf45a81518cc6bbedb003afd7dc4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64, 256, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dae92705f477d9e3f41daf96fe5eb717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b3d2cf45a81518cc6bbedb003afd7dc4
        def get_inputs(self):
            return [
                paddle.uniform([64, 256, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fbf448bf1fbb4fc8ce4e20d6ac14bc52(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 784, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f548d33596b43ee49f2ae7a5d20e12ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbf448bf1fbb4fc8ce4e20d6ac14bc52
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ded5252bf2f246e722d02fe337bf5e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0df2cc75ccd6a284b012797234fdc69
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c5dc33a108a2063622f4f59f9c08476f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_adbcda5feb89fc3698c958285002177c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c5dc33a108a2063622f4f59f9c08476f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0d28b91bec0a6878851c819881ae46e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_06f9129d3cf9b7154815e3b5917a5678
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04e9020230a517b525674f3227d31087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c8c4888433f8726b788878e68e7f778
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_14271f945ff7ff6f9d178fc68cd794da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 512, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8824a363b0a5ccc13ae00270ea887ae2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14271f945ff7ff6f9d178fc68cd794da
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b58c1b474a98c4e3c97bb89e64a17292(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 3136, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cc4b7be125891d3667da5ddcf98504e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58c1b474a98c4e3c97bb89e64a17292
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4aab1fb3a5cc05dcd99de3963c0c36f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b447b78dd7c3749bf5fba3687e5af8
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_790cfe675a3814c546174a90db48003d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 512, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fd4522febac154fddc1d9e94bd724d52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_790cfe675a3814c546174a90db48003d
        def get_inputs(self):
            return [
                paddle.uniform([8, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d7b1a49528649bc175c5bd69a42e9fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f79df13980bb3ddf78cad467c8eef448
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc256cd662f39d893df539e850f5ec9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 60800, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f55ea6be0541228eb68184d394bac98(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc256cd662f39d893df539e850f5ec9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f0d211fa6b1dce76e5aa80e963ce50fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 640, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4e1cd94c28d33ef40ad624b5824b574(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0d211fa6b1dce76e5aa80e963ce50fd
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_990dc91bbd1dddd5cfa29598b5185f4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 198, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3491bd21c4ec20e0b9c6de0a0a395717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_990dc91bbd1dddd5cfa29598b5185f4b
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f75f8432af58ceee74524ebd4b1cfd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_657b7f505cd63962b31f18d8d9699016
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e8a52a93db6f03fd007c0d90cf59d4c1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 169, 1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024], dtype='float32'),
                paddle.static.InputSpec(shape=[1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_04e1d432eefcef4fb0fa6b9965341e70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8a52a93db6f03fd007c0d90cf59d4c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04e1d432eefcef4fb0fa6b9965341e70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e8a52a93db6f03fd007c0d90cf59d4c1
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2c5e6c682936587403418b03ea662d6e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 49, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e468d907dc67b7d1cbaf5bf9367add68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c5e6c682936587403418b03ea662d6e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a7fb37158a55f168389578707318c2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 65536, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2d59a91d302dea9222a32bdd3dec1099(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a7fb37158a55f168389578707318c2a
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f63cb7b0da5ba85af0742cd6f2757180(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 9216, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_389b534ae23f58ff7be3d89b2159c308(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f63cb7b0da5ba85af0742cd6f2757180
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b24bb54e2ff2b1ef6b548b346f75a427(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_266c6f454e0860c752cf3aee447d67d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b24bb54e2ff2b1ef6b548b346f75a427
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_364fe1760d0e9ed25fb1dff166bcea6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4665d4980b92f5773b6f0b5a401b5ef
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_762906817052d989a8b48a47938e0ef7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe0f0267aef87aa4f5f66e1c7dcee096
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_eb5f2705ea9df5e8b1b06c6aae3cfbf9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 144, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_83f9180c8fcda077c03cb46f621c4bb6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f2705ea9df5e8b1b06c6aae3cfbf9
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cc4b7be125891d3667da5ddcf98504e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58c1b474a98c4e3c97bb89e64a17292
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_846b1b7a6c1b467ac5cf04afc70c347a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 784, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_713287a59d4357648096ce230dbe1b61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_846b1b7a6c1b467ac5cf04afc70c347a
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_713287a59d4357648096ce230dbe1b61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_846b1b7a6c1b467ac5cf04afc70c347a
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ade07a092dea05bc9f89bbd3f451b55f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1025, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c97379b634c3fa508144be499ce1cea4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ade07a092dea05bc9f89bbd3f451b55f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cc4b7be125891d3667da5ddcf98504e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58c1b474a98c4e3c97bb89e64a17292
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b8b23eaf16a1d217910420966a4ba898(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 576, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3119e6e44c90b9feeeae130e98e1ad66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b8b23eaf16a1d217910420966a4ba898
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44f55890cb924acafa3504feef6a42ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3405df2db121487fbd00faf3656a32f
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ba5deec6f1191df92afb0584a10cfa9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_17171dd227e86df214e2c95b6bf99361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ba5deec6f1191df92afb0584a10cfa9e
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9e86b635dbc51271fb5d825b7c3178c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28a747916be96985499781ff7fc2c0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2b5d87e3b9df213b282c524ac4299c73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a3419eb8804ae711415a7589775e98d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b5d87e3b9df213b282c524ac4299c73
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c44c1cc247bcc2e936aae3f0b4fdfaa6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0fdca6a8784e7c6e914cbde9c67251f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c44c1cc247bcc2e936aae3f0b4fdfaa6
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65daf24669abc61df512f9b9ffa31b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3f47948f259d9de1a32baa466d8d87
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_517d51f756281cafb82b7256d99f3fae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16, 512, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_32304c2a4cf904290a92ee62f2bc48eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_517d51f756281cafb82b7256d99f3fae
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ec564384040570ae29d07bc51056a1cb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 100, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e00abbf5d4db8ea43bc1d1952a231dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ec564384040570ae29d07bc51056a1cb
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e8ee9f97a48940598814a14337b75cc5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdc7b4d17071c467a0d1e332a6348d00
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fbaae853a4f1d70506e5b783964ed5f5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[54, 197, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9368633cb94902229d0f35f7c632ece2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbaae853a4f1d70506e5b783964ed5f5
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c00e53407e7f89edbedea5fdb3b78768(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 32768, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03f40d3b113ff41904d4867e4cfc6ffd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c00e53407e7f89edbedea5fdb3b78768
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7f40a3ea3a11002a44b59a399756bfff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9a18a06484008bcccbd06eec465bcc0c
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c0e710b9ab566670c00d5ff40c06be8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_007e6e75b717bd959110ef76b6b94bda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c0e710b9ab566670c00d5ff40c06be8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a4901f03e5668c4abeba0e7b817aaa60(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 320, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d39d4829c1609b20959ab2f70142ef73(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a4901f03e5668c4abeba0e7b817aaa60
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cd86b3c8a820a905fc6daaa1ac80d7aa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 2304, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af3bf97fb957aea3e285d8e9422210e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cd86b3c8a820a905fc6daaa1ac80d7aa
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e468d907dc67b7d1cbaf5bf9367add68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c5e6c682936587403418b03ea662d6e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f0aa531bcfd7dfc46bab0631a2647a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52ad6ec3904cc074ee046fc8a3ea60ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c000b6f8f5ea85231d5d635b3007e19b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc8403ef07f461467f322af55d1e1ffc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c000b6f8f5ea85231d5d635b3007e19b
        def get_inputs(self):
            return [
                paddle.uniform([8, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ecec78218a5b480dab4c240f6b43491a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 196, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f5b77e32d0d6973d31b7e673ae874100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecec78218a5b480dab4c240f6b43491a
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f75f8432af58ceee74524ebd4b1cfd39(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_657b7f505cd63962b31f18d8d9699016
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_12958c86117251a1e0e0895ac800973d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 3136, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38eb56355a639fdd71577fb666b6a5b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12958c86117251a1e0e0895ac800973d
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9cbe492a8ca448e5f4e72deeaba6776a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9cd1d794992abe22bd83eceb48021dcd
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_04e9020230a517b525674f3227d31087(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4c8c4888433f8726b788878e68e7f778
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a6e67fb9dfc928caa04c72676e2bafb7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_216f69ad64d1fce2243840635140d282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e67fb9dfc928caa04c72676e2bafb7
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2db86ffc69b0322526c584ff7ded4e77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 200, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6dc7f37c837b1384aada75c3435b4ec7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2db86ffc69b0322526c584ff7ded4e77
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dbeef597ef9803e346a77c5f959158d6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 144, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f8de61565243f6e3da08f235392b885f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbeef597ef9803e346a77c5f959158d6
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e468d907dc67b7d1cbaf5bf9367add68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c5e6c682936587403418b03ea662d6e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_65cd11afef66538d6f2354158ea12a90(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d3a2c887ce0bec2b58dd43ea52e4f754(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_65cd11afef66538d6f2354158ea12a90
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_99e55c5b9a4d9eec93fb34f489c2f059(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 21760, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f27f702202d2f8fee2ed0f9751e7d02(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_99e55c5b9a4d9eec93fb34f489c2f059
        def get_inputs(self):
            return [
                paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_713287a59d4357648096ce230dbe1b61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_846b1b7a6c1b467ac5cf04afc70c347a
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9616b1757c32e3bf9820740c40a8be1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_457692c642c942c0313d56884d84bc1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_12800ed4544d977fa4dfbf3526d870ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
                paddle.static.InputSpec(shape=[128], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_31eac8c0ec00774709cd5193feb1c57d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12800ed4544d977fa4dfbf3526d870ce
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e826f2679dc8c0eaf5f2ec7905f5a2a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549efa21d993482520c268e8e8a35987
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_59b562a9bd80304e74b2a12ff7b1cdd2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4db516bc602e6941016fb340e844f6b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b562a9bd80304e74b2a12ff7b1cdd2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b109ca482ffb11f13c548df4825eb504(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 9216, 96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
                paddle.static.InputSpec(shape=[96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6ab6f377bef79bde33e59d3e1d20157f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b109ca482ffb11f13c548df4825eb504
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5b77e32d0d6973d31b7e673ae874100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecec78218a5b480dab4c240f6b43491a
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44f55890cb924acafa3504feef6a42ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3405df2db121487fbd00faf3656a32f
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6ab6f377bef79bde33e59d3e1d20157f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b109ca482ffb11f13c548df4825eb504
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e9e3320d6f812d7e10bf6ed83c5b5604(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1025, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd7bf4548cd8e9fde9463b5a73f33fa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e9e3320d6f812d7e10bf6ed83c5b5604
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d7b1a49528649bc175c5bd69a42e9fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f79df13980bb3ddf78cad467c8eef448
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38eb56355a639fdd71577fb666b6a5b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12958c86117251a1e0e0895ac800973d
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44f55890cb924acafa3504feef6a42ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3405df2db121487fbd00faf3656a32f
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ccf7abaac7d7ab689d2a73023c625453(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ae7a4dc4a1ada57234bd794bedc0d9db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ccf7abaac7d7ab689d2a73023c625453
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d7b1a49528649bc175c5bd69a42e9fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f79df13980bb3ddf78cad467c8eef448
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_389b534ae23f58ff7be3d89b2159c308(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f63cb7b0da5ba85af0742cd6f2757180
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e468d907dc67b7d1cbaf5bf9367add68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2c5e6c682936587403418b03ea662d6e
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aba87c5d9c47243014698e1e78459d65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae023a3d25c30e6782582b8a0b634724
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c354f1091dea000db5cc8c6079b0a25c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
                paddle.static.InputSpec(shape=[320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77cdf23d28cc51c8da1da925335c37f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c354f1091dea000db5cc8c6079b0a25c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f0aa531bcfd7dfc46bab0631a2647a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_52ad6ec3904cc074ee046fc8a3ea60ea
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3c3ac965038c3aa161030a93e31c7d9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
                paddle.static.InputSpec(shape=[160], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10af7890daea00c0fa0eddd2ad28b0bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3c3ac965038c3aa161030a93e31c7d9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8090323cbe08529fc57477e2a9827153(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7edc30136aa5d1f66aeb085fbdd31b02
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_47920d55abf47f1cf8771daa54b67941(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3590263ec51887bd0043defcf7d39a9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_47920d55abf47f1cf8771daa54b67941
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f548d33596b43ee49f2ae7a5d20e12ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbf448bf1fbb4fc8ce4e20d6ac14bc52
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_af8697ddb18841e42f355b85d04d1b80(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 2304, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ccf8a42d63c556b3a3e179abdf0decf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_af8697ddb18841e42f355b85d04d1b80
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_615d5089923064b76a250151c39515ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b2f097e060e3056648efa4523955b7a9
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5b77e32d0d6973d31b7e673ae874100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecec78218a5b480dab4c240f6b43491a
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_364fe1760d0e9ed25fb1dff166bcea6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4665d4980b92f5773b6f0b5a401b5ef
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f548d33596b43ee49f2ae7a5d20e12ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbf448bf1fbb4fc8ce4e20d6ac14bc52
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cc4b7be125891d3667da5ddcf98504e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b58c1b474a98c4e3c97bb89e64a17292
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d3abebd9f8aa4c4338dae05dc6ca11be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 576, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d92acfc62d596747bd34fe4dbc610a63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d3abebd9f8aa4c4338dae05dc6ca11be
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4aab1fb3a5cc05dcd99de3963c0c36f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7b447b78dd7c3749bf5fba3687e5af8
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b259e1a0bbe4c580b17d80ffb44224e9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1024, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
                paddle.static.InputSpec(shape=[64], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3e5003592b7190e0d884947efd7e1a0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b259e1a0bbe4c580b17d80ffb44224e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe46e529872c86b28b39c9d6d27356cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fcb5ebc91a181687d19cfb7c421a310d
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_216f69ad64d1fce2243840635140d282(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a6e67fb9dfc928caa04c72676e2bafb7
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aba87c5d9c47243014698e1e78459d65(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae023a3d25c30e6782582b8a0b634724
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_15ee7c769a4c26aafc12601f89dc0ef4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 256, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_28fa236d0e484103eed528cb399aadff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_15ee7c769a4c26aafc12601f89dc0ef4
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aab0d9c3c756f034c1052470b2426876(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[86, 197, 192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
                paddle.static.InputSpec(shape=[192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_86fdb659a54e9e5aac85f9fb9a14d142(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aab0d9c3c756f034c1052470b2426876
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65daf24669abc61df512f9b9ffa31b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0e3f47948f259d9de1a32baa466d8d87
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_a7f67675400569a3595a218823e26474(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
                paddle.static.InputSpec(shape=[32], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_482e8093ecd20ff355dae72d894f526c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a7f67675400569a3595a218823e26474
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e826f2679dc8c0eaf5f2ec7905f5a2a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_549efa21d993482520c268e8e8a35987
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07bb44290efc6cc6771b279b82325dd5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 128, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
                paddle.static.InputSpec(shape=[512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d9da3904b77cccc6776593102b107377(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07bb44290efc6cc6771b279b82325dd5
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f5b77e32d0d6973d31b7e673ae874100(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ecec78218a5b480dab4c240f6b43491a
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc5e718a7d4a108dc0b390f7254f33ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 160, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9575712466053c26090ff1a776f14036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc5e718a7d4a108dc0b390f7254f33ff
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cc01492fca0ba51989fe794a4bf95427(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1174, 384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
                paddle.static.InputSpec(shape=[384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b139bf32335cda94458e0d2e0889e461(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cc01492fca0ba51989fe794a4bf95427
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9e86b635dbc51271fb5d825b7c3178c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e28a747916be96985499781ff7fc2c0d
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e94763f0abb555fd37f7f4ce34a9556a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[4, 128, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be2e2719cd1924d35f11bba6598e26f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e94763f0abb555fd37f7f4ce34a9556a
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_07c1744acac6ee6f10464bb83a795977(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1174, 768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
                paddle.static.InputSpec(shape=[768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fde87d3c4a7d7d80975b997ca50c321d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_07c1744acac6ee6f10464bb83a795977
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cf20ed8cc59816fdff31853fc36bb034(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_18b92ebe5c2c2168533c86b1a8db4c20
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e5003592b7190e0d884947efd7e1a0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b259e1a0bbe4c580b17d80ffb44224e9
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38eb56355a639fdd71577fb666b6a5b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12958c86117251a1e0e0895ac800973d
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38eb56355a639fdd71577fb666b6a5b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_12958c86117251a1e0e0895ac800973d
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_26ede4b22edb88c4bffdf1eabb80e296(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10, 50, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
                paddle.static.InputSpec(shape=[256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_178bd618e09845c24ad9ba7ce35e0232(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26ede4b22edb88c4bffdf1eabb80e296
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-05, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8eeae09003176de2a0f2af94c58ef1a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1, input_2):
            return paddle._C_ops.layer_norm(input_0, input_1, input_2, 1e-06, 2), None, None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_101d067cec4ae7621952debc1cf7620f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_98f6f4956a62ac008905fd4542926ce4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([54, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20c3b62b31a45e182d77e7173bcca4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df1c54f28fc915520fd8d1b733b9bdb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fdfaa871a283c82af790e6e3e99ec1df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([16, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8744e82f2b9bdd9f1f34d5529fa84b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e80fca5f29956ffdef31bca804d6fdbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ea9445b601c37b7cd75808a2cde18b4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([128, 32, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_216d269c97129032fa606b968dacf89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8cd412787328a04f19931a77d9a9e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7757bdc4fb806e11490c4af6bb1865e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fc106a01e67ca3aa8fe0fb9dc6fc9fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_babecc23bffa5c4d1805a3c6d34d8f72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a03563b53c6197d6feb8d4574e5bc4a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f017b6498cd28aea547e03e1179f414c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f728ceb5c2b39d82b021a704bd2a780f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c84259c1752cc4fdfebb0f8d7efd3a94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa09b29f511b22ca04a1d4be5a007b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca1ed945b3c1d2a70e0576ce816834c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([64, 256, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4a0c34cbc747c8ba27472cfc5b50680(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e80fca5f29956ffdef31bca804d6fdbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8e702d08ace4d36944f10313e32c472a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8cd412787328a04f19931a77d9a9e93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3395d9eea881b6ce6a334bbd7395cacd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_095c6a3bda89630d017ceb2ebc68b50f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38491dbf235d5f2decb80f8a9b83737f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbf888bc59b8cbf95f17ad7002da1466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fcdc73794d6a68768581e7d406c6738(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([8, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_216d269c97129032fa606b968dacf89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d2ed7a927ed3a2373131123cd965c635(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 60800, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bec64f03a5f4019e90b0d88beb140250(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([10, 640, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7cbed586e0f038c55c892f01cd9d1472(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([86, 198, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80804a988ebb525171eb79bacd8a09f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ed4bf82d311d5cbe65a16c18177c036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5ed4bf82d311d5cbe65a16c18177c036(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
                paddle.uniform([1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b9cd29560f9a01cf6e7cb9710b551f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6006ada8f02b40f9cfa74087561a4b28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca0ff7b7ba130dde4e2eb70584479804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0dff5d3e65dde99bfc9845577ff05126(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_313cdb34cf9047feca79656a74260da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_101d067cec4ae7621952debc1cf7620f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb783e8b8148ad2be7be2f0e8aca3b29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([4, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38491dbf235d5f2decb80f8a9b83737f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_342e405d794694f1ae11b21c74012194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_342e405d794694f1ae11b21c74012194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d22e45ed24e465e87208f726869db999(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38491dbf235d5f2decb80f8a9b83737f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2b8da1be900c3fa9dd63a30b70dbddb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([4, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c84259c1752cc4fdfebb0f8d7efd3a94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_656a84e3535dbd256f554ef55b49b05a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f890bc1a75fd48c377467b0c1bf72b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a700a747cc8080d83e789acda3f6c658(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_058e2fee335f8a9aa1dc77b6d5a5f87c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d092f0fa99cdf48aa87852d91e99a213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15937475f2e54805edaf620321563128(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([16, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d7f037aa7a6310316a79eec002b9347c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([10, 100, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8eeae09003176de2a0f2af94c58ef1a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b09e60ebfbc6d230e8e147a1196d2a06(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([54, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37b094eb0a531b2688acbfaaec921271(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa09b29f511b22ca04a1d4be5a007b8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_330142f7be2ec586b3bdcbea12f57ad3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b1e9378ae8adfb9ffa762418a6ffd384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_aa79270f1e092166180e540bf49955af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([4, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b9cd29560f9a01cf6e7cb9710b551f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9f7160fcf6dab561283e205c6959d7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afa1901f3051d16a056716e5870a0f1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([8, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_262a22b8784f64abccd9c6d41f3fb007(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80804a988ebb525171eb79bacd8a09f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_780f59c568cfd294d34d8e90d8d45a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7757bdc4fb806e11490c4af6bb1865e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([10, 320, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3395d9eea881b6ce6a334bbd7395cacd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3eafd3e7833e6772a49edc30bd43e8d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7846b5770fac57899c43e0d10b1a6265(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([10, 200, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_62e715862ac6b67b86006cb095052381(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([6, 144, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b9cd29560f9a01cf6e7cb9710b551f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6834de5cef348c1c9c773a772a735ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86290f527fe35b8963a08359cf0a0cad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 21760, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_342e405d794694f1ae11b21c74012194(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fc106a01e67ca3aa8fe0fb9dc6fc9fd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a52a33ceee3e7c00ba1d500f33d7e942(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
                paddle.uniform([128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d5bcf7c71620145c2bd6fa12629cb7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cab8f30b07e59eb2e151917bf6cf544a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8176f74c7ab3e7d1e20b85dd06aaf67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_262a22b8784f64abccd9c6d41f3fb007(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c84259c1752cc4fdfebb0f8d7efd3a94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c8176f74c7ab3e7d1e20b85dd06aaf67(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([4, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0f9e7e1eb5cf11c5d46dd9baa2fe268(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_216d269c97129032fa606b968dacf89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_780f59c568cfd294d34d8e90d8d45a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c84259c1752cc4fdfebb0f8d7efd3a94(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4447eb0db1b8785424ed888cee73e75d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_216d269c97129032fa606b968dacf89b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ca0ff7b7ba130dde4e2eb70584479804(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([6, 9216, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b9cd29560f9a01cf6e7cb9710b551f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff37846baa6ae1d2682417171564f4bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d01ed3c8b88e4d386c77f0c1abc7bfdf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9f7160fcf6dab561283e205c6959d7d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9497b1cb6b1457bdafa86ef36b23ef53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8744e82f2b9bdd9f1f34d5529fa84b54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a5575fbb922bbed69c24dfd1d0e88e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4a0c34cbc747c8ba27472cfc5b50680(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16741521c3b23b5116643f50d35427d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([6, 2304, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_20c3b62b31a45e182d77e7173bcca4f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_262a22b8784f64abccd9c6d41f3fb007(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_313cdb34cf9047feca79656a74260da8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4a0c34cbc747c8ba27472cfc5b50680(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_38491dbf235d5f2decb80f8a9b83737f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4aa7fb7f110d3a03a42bcd0c9b476af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([6, 576, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbf888bc59b8cbf95f17ad7002da1466(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 16384, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3f1b30b9fdf804c820d5aeaff7f4cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a03563b53c6197d6feb8d4574e5bc4a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3eafd3e7833e6772a49edc30bd43e8d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff37846baa6ae1d2682417171564f4bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_37addf99b67f3522180558b6ffc6df32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90ac90f2c21bc93d42dbfa17e2672209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([86, 197, 192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
                paddle.uniform([192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d092f0fa99cdf48aa87852d91e99a213(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 32768, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3b014ea13b6ad93543775d47606791d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
                paddle.uniform([32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d5bcf7c71620145c2bd6fa12629cb7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
                paddle.uniform([320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_333bac19c258595d4b02710d2d62ca5b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
                paddle.uniform([512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_262a22b8784f64abccd9c6d41f3fb007(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24bc10f2cb04f998bc065149f33aa624(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([10, 160, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9784378ca0af97b2299fea675f955af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
                paddle.uniform([384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f890bc1a75fd48c377467b0c1bf72b20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
                paddle.uniform([160], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_76d9d03c3e991cc30ac12fc54c89d796(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([4, 128, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afeed82d738c3bfeb1632f25f3b9b5bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
                paddle.uniform([768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_df1c54f28fc915520fd8d1b733b9bdb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([1, 65536, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3f1b30b9fdf804c820d5aeaff7f4cc0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
                paddle.uniform([64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_780f59c568cfd294d34d8e90d8d45a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_780f59c568cfd294d34d8e90d8d45a6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_72255a24c1760cfc7acc8d19f2bc2064
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
                paddle.uniform([96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f220e56b16c5f65bc7c357b86e8e0e78(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e001e0700c3cde6f5ba160b749f42f4c
        def get_inputs(self):
            return [
                paddle.uniform([10, 50, 256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
                paddle.uniform([256], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()