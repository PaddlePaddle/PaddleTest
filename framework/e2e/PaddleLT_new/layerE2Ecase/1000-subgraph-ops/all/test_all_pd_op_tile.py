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
    class PrimitiveOp_46993fafb9d37daf5b126ad8e0e6bebc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 300, 256], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e927c94a7efbf701cd5445b90d14f972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46993fafb9d37daf5b126ad8e0e6bebc
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b19b0767f7b7c9f5d32ea5766c0f935f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a8a90706c8b9fedd2cf148375ca64de1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_60061f16c2ffdd388a9f5f2426a0379b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_177504340edc59aea1453fcfdbc13e6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_b19b0767f7b7c9f5d32ea5766c0f935f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_a01d9e88c31086d4e78721290cbff3f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 76]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_44a9d9812ffbf883dae3dc2f1f49a141(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a01d9e88c31086d4e78721290cbff3f0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 76], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_2435cdaff3acbc5b0b2dffba7cbd6bf4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_a26b6b0dc838605358d44d0232512bd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_39c4dc1eeb88d7f7bd445a108da5708d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_dc0c2aaf7604b42b6e32864d21db9aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_d8d517e812b1a364dd018e1c6901fe9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_175a38f0f1d7e327280ac5f257dd4b68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_c390e6177dddd874585a86adee6b9131(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_d0c69f27548177baaf2b81367e66d059(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_92b718c38de158e7ea051bdf6d5844e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 100, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3f9955b977405e11bfaa5102a46e9174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92b718c38de158e7ea051bdf6d5844e5
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.1778058111667633, 0.3185793161392212, 0.25147324800491333, 0.41976305842399597]]], dtype='float32').reshape([1, 1, 4]),
                paddle.to_tensor([1, 100, 1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_32325f471068ffe253bab4067d98776d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 300, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6d77585534c4f732582081355e9b857f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32325f471068ffe253bab4067d98776d
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.002517612185329199, 0.07827232033014297, 0.10384435206651688, 0.3168522119522095]]], dtype='float32').reshape([1, 1, 4]),
                paddle.to_tensor([1, 300, 1], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_5eef08918d5340d1f2cd9654aec6ea1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_6c9ce5267c248fc0855246324a9f900f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_4830451af67b2a73c082ef489d1c4536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_72fb4979351565d487c9044fec5a787c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_3a36c15ccc3ff941a0593775c5d5b4fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_8783c91b4899b29e2c12ccaecbeadf90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_b4e5f14c5850b6db63f44e9511a13812(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 512]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_234914a429f1503ccc9f2fe36962a6c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e5f14c5850b6db63f44e9511a13812
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_39c4dc1eeb88d7f7bd445a108da5708d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_dc0c2aaf7604b42b6e32864d21db9aab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_234914a429f1503ccc9f2fe36962a6c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b4e5f14c5850b6db63f44e9511a13812
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_58d6899007b85cda956d0f39f3456e3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d9da8ceba5b698740ff64ab7815cbb5
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_57bbb81f215a9cc499d1b316b4596488(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0e32b8a8ac6cd5c02eecc84f3551c73
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_e927c94a7efbf701cd5445b90d14f972(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46993fafb9d37daf5b126ad8e0e6bebc
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_216d442065f808d011a2014677fd762b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68583f1fadbdbfabe940197e1c4e356c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_216d442065f808d011a2014677fd762b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_5977da549a3917c21df8bb25d8384a9f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed7e0a6979b200561985c93a5ff5779c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5977da549a3917c21df8bb25d8384a9f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_369463506f583cb839d26d0e1e2f02df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f3378c58c185996f778d214bbe637f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_369463506f583cb839d26d0e1e2f02df
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_d2c56f44024259a3ad00ff1d5c088cfc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 11109, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bbe8cf49a90b125a027140c0b8568e7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d2c56f44024259a3ad00ff1d5c088cfc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_68583f1fadbdbfabe940197e1c4e356c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_216d442065f808d011a2014677fd762b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_1d795eac953d9b58fa4f6999777f3e0e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 76]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6976681a6c28b6d91828a974eb2a384b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1d795eac953d9b58fa4f6999777f3e0e
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 76], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_fb32c9eeb9858bb5d834cd8d2c228810(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1b4d6f7b6688fead48084aa9cfe6d507(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb32c9eeb9858bb5d834cd8d2c228810
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_92db4e98d8f2802d9ac07a3537d91d63(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3024, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4572671e8034419c7ef105e47188b9e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92db4e98d8f2802d9ac07a3537d91d63
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_55da94ed497478735716945ffb6b4f2d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_82050d7991628e35186b50358f1d29d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55da94ed497478735716945ffb6b4f2d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_81dba919628e45304c6736cf82dda53f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_71d4cb01582a4bf063dd32bd029b8f63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81dba919628e45304c6736cf82dda53f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_93801a117bc7aab01a9afe3fa70aba4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad8cdffa979c577007b1f80f8cef028b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_93801a117bc7aab01a9afe3fa70aba4b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_81ec6cd55dea9e8dc8004a3827311512(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 9261, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0545fd1ab141e41cd305ba25e2825ef9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81ec6cd55dea9e8dc8004a3827311512
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_83172d6d490dd96f3bef518747282c3e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b792790a929e5a06de8586b0c54a9a1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83172d6d490dd96f3bef518747282c3e
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_0b30c1a9a320125b9559aa14e2ee1d97(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_abca82b733712aa05145228fcbd1765c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0b30c1a9a320125b9559aa14e2ee1d97
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_38a155518dd2af829c494e7a38cd9b64(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 100, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_77146f2cf3bb070839f7a4d8f674a391(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38a155518dd2af829c494e7a38cd9b64
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.1778058111667633, 0.3185793161392212, 0.25147324800491333, 0.41976305842399597]]], dtype='float32').reshape([1, 1, 4]),
                paddle.to_tensor([1, 100, 1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_9eb24591a1b0c2d0241f745a15bfec95(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 300, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, 4], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ac012ab6a13ad0a419c9a3be86a80699(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9eb24591a1b0c2d0241f745a15bfec95
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.002517612185329199, 0.07827232033014297, 0.10384435206651688, 0.3168522119522095]]], dtype='float32').reshape([1, 1, 4]),
                paddle.to_tensor([1, 300, 1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_38366de6b59984b5f0a61a356c10e327(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aac0f512a42344a47793142841a23ec0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_38366de6b59984b5f0a61a356c10e327
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_08a0b70c6a84072c2c5d844fcd600524(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4725, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36c26f264be8a33d46fe503cb5737ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_08a0b70c6a84072c2c5d844fcd600524
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_d067b9c311e56fd1cce23358758306ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3bb347079d7a34168ae48af34f3b7c32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d067b9c311e56fd1cce23358758306ee
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_005ec9630d7f9830d62c301fb0cf5cef(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e613e2f5932ef63b791732ada004f1f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_005ec9630d7f9830d62c301fb0cf5cef
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_5b309b668d43957827f26fb1b41d12b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_690baa3f1e610dda4f2a636a67f9f3de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5b309b668d43957827f26fb1b41d12b9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_c037a381a37c17548698052c7b38b809(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 7581, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ed768a806b3fb5fefdc6d00164284139(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c037a381a37c17548698052c7b38b809
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_4e9acadece78cc971c80f82695e877b2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 512]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 512, 1], dtype='float32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fb7e4404b469787964eac4a4eb7f9dd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e9acadece78cc971c80f82695e877b2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_82050d7991628e35186b50358f1d29d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_55da94ed497478735716945ffb6b4f2d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_71d4cb01582a4bf063dd32bd029b8f63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_81dba919628e45304c6736cf82dda53f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_fb7e4404b469787964eac4a4eb7f9dd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e9acadece78cc971c80f82695e877b2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_930c065eba667c61cdf4ca07aa16be2b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8bb3d169adbdb378365505ecfb4f3c58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_930c065eba667c61cdf4ca07aa16be2b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_d5321ad370e317080b56d9788d28197e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 1], dtype='int32'),
                paddle.static.InputSpec(shape=[3], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_366e72f3b5373cb045f0b8916710f3ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5321ad370e317080b56d9788d28197e
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_6812320dd7db3747c315914b03d3223a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e64889586d87d3089c19a539a6eb8a79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6812320dd7db3747c315914b03d3223a
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8c28c6050273654568eaed4db2d797de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_316cca0422eeed93e4ec82f89820d680(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5d6a89fe771904d730d2d1365213465b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_7b8c7291776707f4a982da3107276464(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_abc66cf53e26a9d0caefa047764deb2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_8c28c6050273654568eaed4db2d797de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_d59776e57a09218b486f5c5a8bbb7a90(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 76]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a7230074c758727438f9b8c58111e24f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d59776e57a09218b486f5c5a8bbb7a90
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 76], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_0a46354f6d21d4491a8a700deb38ea34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_784112ba3da24ee5e6d17835f36d0b38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_2fce3eab6b21688a01feb6008daea329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_0a24c1a2992a3adeaab9320e18b64467(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_897f2620c0202ba4f38e7f64d08916c2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_17f9485b7a6ee6d0a863ea5748b530cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_4419057bb3dddaee8c516960ab591720(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_e7ff6fbc53d5c24e3ea3e23da489c454(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_fd38a16d0358121858a37d517ebada9a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 100, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd0c8a6cf0233c697d3d36b7a517135b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd38a16d0358121858a37d517ebada9a
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.1778058111667633, 0.3185793161392212, 0.25147324800491333, 0.41976305842399597]]], dtype='float32').reshape([1, 1, 4]),
                paddle.to_tensor([1, 100, 1], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_9477351bcb2fa9680a3602fc7551f625(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 300, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c06ca6c00c41a3cc01e7884616663eb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9477351bcb2fa9680a3602fc7551f625
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.002517612185329199, 0.07827232033014297, 0.10384435206651688, 0.3168522119522095]]], dtype='float32').reshape([1, 1, 4]),
                paddle.to_tensor([1, 300, 1], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_90944774a4e26e16a0fc5782d68fb197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_65fd475df052698759e5e4a42ecc4730(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_1f58dbcd9df7502c40407c70e92bb325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_d748b6723ec6b3b0fa3aeb42e0688d64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_47c00592659e7435accd5bf3fa828fa9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_4814a44390a6d8e2595d61f5a349c1d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    
    class PrimitiveOp_c4052ee98c372bcba4039ede3c4ef56c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            input_1 = [1, 1, 512]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_67bef9407f0a3901ebc0e8264a1b0d1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4052ee98c372bcba4039ede3c4ef56c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_2fce3eab6b21688a01feb6008daea329(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_0a24c1a2992a3adeaab9320e18b64467(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_67bef9407f0a3901ebc0e8264a1b0d1b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c4052ee98c372bcba4039ede3c4ef56c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
                paddle.to_tensor([1, 1, 512], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_60b644c2256cc58f80286ac6d3f6646a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e4a7b212e5ded3b6a306c46751775da1
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 4], dtype='int64').reshape([3]),
            ]


    class TestPrimitiveOp_6203eba372c7c91059cc0fc26be90622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_316cca0422eeed93e4ec82f89820d680
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
                paddle.to_tensor([1, 1, 68], dtype='int64').reshape([3]),
            ]


    

if __name__ == '__main__':
    unittest.main()