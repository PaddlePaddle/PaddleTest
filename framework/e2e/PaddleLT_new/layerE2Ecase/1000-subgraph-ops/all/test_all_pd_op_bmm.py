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
    class PrimitiveOp_628d175bf494f1c0cd2a4bc67d59220e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 19, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8aeb92e3f34a01ac7f939e80c65582ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_628d175bf494f1c0cd2a4bc67d59220e
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cf6938aaea1a6598205c26b6d3677791(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 21, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4204a1cfd8fe78bdf4ed8159805a6e83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cf6938aaea1a6598205c26b6d3677791
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d7a7311ce68f8aee4dc096b353974836(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 64], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 64, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1ec68cf4e2440e4e44a5223e30da3f9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7a7311ce68f8aee4dc096b353974836
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_38cfc04ed9dad25d92d656873d60d98e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4f0d8758ffe17b3136747b845ea364b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38cfc04ed9dad25d92d656873d60d98e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a88756900ad172342a5b0ece9c9c7d18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d7a7311ce68f8aee4dc096b353974836
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6b07d57c5d6d8931ad9c45c2cdfa940(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38cfc04ed9dad25d92d656873d60d98e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6ccfe828d0f0854f1689e96f1907689a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_de747f01c569af782a31100f15600e4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ccfe828d0f0854f1689e96f1907689a
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b0990c9bf7c7eccc9289aba1a537e06e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 512], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 512, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dc319f987f078c6018d8c9d0cea1ac61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0990c9bf7c7eccc9289aba1a537e06e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a421699ff2137c5b7eaa97dd51e28ac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6ccfe828d0f0854f1689e96f1907689a
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5d656ef40fd9a727cd685c569186c05e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b0990c9bf7c7eccc9289aba1a537e06e
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5751d621c1ce15257eaba26fa1448901(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.bmm(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d723c3f271c866254bbb9af66a00585f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 19, 32768], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 32768, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4fb73ea039ee1e221982d529798a8d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 21, 16384], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 16384, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_97b4d8f07723a256a787c46d806fdd55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 4096, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_84e6a976bbf15c7526df4ac2e892fd4c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 4096], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_752497a9fad9f1f1bc43255b7eb51f5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 8192, 64], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 64, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7da4a15819b89e264b63425a195fe79a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_519238e4b2ba24c050c2ef5aec0f6ae9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 8192, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9c3cf4f67f36ccd8e9b0ded44763ba4b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 8192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c944e85e1614fab8323f117e7a5b03a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 4096, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2df1b0ab72565279d12b9865fe483064(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5751d621c1ce15257eaba26fa1448901
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
                paddle.uniform([1, 512, 4096], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()