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
    class PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_ed57fc976b74c7264c41a2ae1d62b306(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dbd46434a3d4d6668eefa389a8c47a89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed57fc976b74c7264c41a2ae1d62b306
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_49b01bf697fc4aead173e5def1358871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    
    class PrimitiveOp_68694b970158fd2dd9d6df5119b46364(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_414d6fe46a236283c65b74a3b1087e4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_a9d178494b67885762dbad6e58b22c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_b422abe97bc37e08f8628e492710b4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_bc4f829001b27144f1e99a9169ab8ea9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c85ee3e36c3aaaa5359c6026a40ed384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc4f829001b27144f1e99a9169ab8ea9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_7020f85e47c4842fd33575860d3b2abd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed57fc976b74c7264c41a2ae1d62b306
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_8419fe993f498f2a363e70169ba064d4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ca76330bd662f8046df8d19ef298e45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ca76330bd662f8046df8d19ef298e45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_c7f2ce5826cccb329cb51508bb2cdb1b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2100], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f73b2ee997e41a54a7467dc078edd808(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c7f2ce5826cccb329cb51508bb2cdb1b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    
    class PrimitiveOp_e0a3c36447721b7b29c0a891d401339b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7188518a54b3c35114bf9384fd0fcc5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a3c36447721b7b29c0a891d401339b
        def get_inputs(self):
            return [
                paddle.to_tensor([128], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_3e16c5db15aa43f4f80d0311e37f59e6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a3c36447721b7b29c0a891d401339b
        def get_inputs(self):
            return [
                paddle.to_tensor([16], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7ab89a3b6df600838058b16415c8d26c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a3c36447721b7b29c0a891d401339b
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_6b7e85cf583f87e39b4b73cf8d39ff7b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[96], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_759730b8507ed4860cb1736648d782d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6b7e85cf583f87e39b4b73cf8d39ff7b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[96], dtype='int64'),
            ]


    
    class PrimitiveOp_783730f1146256ae84f6e2f27f411741(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[48], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d6022bd87ad9ab107ec7a59a9bcc02fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_783730f1146256ae84f6e2f27f411741
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[48], dtype='int64'),
            ]


    
    class PrimitiveOp_9c33769289e3d706da171ca0d6a26650(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[24], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc2a102dbda9952ba6b8afbccdc96f43(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c33769289e3d706da171ca0d6a26650
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64').reshape([24]),
            ]


    
    class PrimitiveOp_6bf5cc315b8e2592762a204be0d57d02(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[12096, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6fd59c4c9beb701b22e9c9b1bf5166b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bf5cc315b8e2592762a204be0d57d02
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fd59c4c9beb701b22e9c9b1bf5166b1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6bf5cc315b8e2592762a204be0d57d02
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_c43d3f1ea3667c3516ff454fed853da5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_290755d51d43534f7521f66b2a924325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c43d3f1ea3667c3516ff454fed853da5
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    
    class PrimitiveOp_560d5c5253615d2d02c631e63e63da2f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c9c515a20a0210e99220dcd0c095e76d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d3977de51310b5b675e68ae607bed020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d3977de51310b5b675e68ae607bed020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_c865ae5640000cc53f2a53c8096377a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cca141ea5a2d6760c7232b0f07517ec9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37460729479789734, 0.4844105839729309, 0.181888610124588, 0.44207683205604553, 0.02534153312444687, 0.3211316466331482, 0.009519957937300205, 0.40302348136901855, 0.22982287406921387, 0.10425077378749847, 0.3179247975349426, 0.40923941135406494, 0.2358681559562683, 0.14795830845832825, 0.06801445782184601, 0.4440118670463562], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_5b22f73a91049458cc542f6eec4653df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_5063f03bbc4e5647c34bec6cab92fadc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_f8a99e88f8e25d2ac557ce3e4611ea68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(7581, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1f3aa8b7105bd71844a95c1b689f72d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_157dd9fa55e0486bab7f9bddba2c27a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4725, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_450d8a9dafa9e6e3084f554adb7cf93d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc4f829001b27144f1e99a9169ab8ea9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_950609f8452d9f93dc5e0b9905db03f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(577, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2fa0394b232a2d711df9c8cdf05f3908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f1b119b493ee43bb642656577972ac2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d0e44ec36e55816d9a40120b44708842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_a91f1d3030b648195fd93cca47c43d57(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 4], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8bf49f287ffe1f800204cd3227bd9079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2b70fff76ea2884eeffe289b4931a7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_2362317998c47fffe52d05887d08c74c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 68], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_460e435e240adb08e41dbba36072e77a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12166afc0710dd781101b27682b99d54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_01b0f4176a77c4db88483fe96504a35f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_86bc7efb550c8f89e23781b8eea1b4ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_48981f5deab36851a461f8586d0e4e20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7f69b8259cd3844f5cceb36e6e24b709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8400, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_b5def456afc3c477ed053ee8d72f2543(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[64], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c66475fcdb037542eb2e675ab53f45e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b5def456afc3c477ed053ee8d72f2543
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    
    class PrimitiveOp_34475c8fc37621dbeed64fca94f23a0a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[32], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2f6920bb55f0c76940d0877745effc47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_34475c8fc37621dbeed64fca94f23a0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    
    class PrimitiveOp_8814cfc92bcc33cb941acd53be2d5470(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[16], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_646ac94b94983b9557aab50881edc9b0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8814cfc92bcc33cb941acd53be2d5470
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    
    class PrimitiveOp_c2698c60825b709d7f50d55aaff5008a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[5376, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a35b898c050bd417f56a4a80e303cbac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2698c60825b709d7f50d55aaff5008a
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a35b898c050bd417f56a4a80e303cbac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2698c60825b709d7f50d55aaff5008a
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_49b01bf697fc4aead173e5def1358871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e852d794f2f351c90a94db0cf6f8adb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7e93c62262971dd31181119b7f7cc177(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_06b32c084977cb4acd8c3dbe994c8c5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_913553049f8c6be10283b55f4616aa83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_567e988fb564cbec8e18f18fac23df18(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a95a0e2638c1fb27e010f5e7ef92f96d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_d71264d7be79d8ccb36ce5a17381c70f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57fa103716ac5f57d4c3f79881cc070f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e238e4e14133595de0679e12bfaca88c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0da1b6c314859a47b30fa02fd594f9cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e238e4e14133595de0679e12bfaca88c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c72eb712919297e6293b36658e40d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8e2b45b31a7cca0e516d18f74aafb297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_efe176d135affaf53c74fccc1e39ff64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_be4563c484a7b87c1c0277fb3d46af0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(98, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_fb7c70c6811d3b4fa6c85c198a8604b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(99, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_708a44f625e5732f7fb086b8ba1f1c38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980c19d67909f8c4d939dccc9ff26d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_980c19d67909f8c4d939dccc9ff26d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_818799804309fbf751b3924eeab0f59c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(192, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_89674e2b4c26e42633059b27664cbcfa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0360ce268380663bd42243199c18915a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_89674e2b4c26e42633059b27664cbcfa
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bbe4bfe5246bd8560e2aa26f839f4d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 192, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8a77584d23b9d5477d6052959d54b0bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_1f3aa8b7105bd71844a95c1b689f72d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6e852d794f2f351c90a94db0cf6f8adb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e93c62262971dd31181119b7f7cc177(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8e2b45b31a7cca0e516d18f74aafb297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_494d4ee3688a3d2fb8a99bb607cfdaf3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2569206b57261dea2d17cfe96bcb1e0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_494d4ee3688a3d2fb8a99bb607cfdaf3
        def get_inputs(self):
            return [
                paddle.to_tensor([False, False, False, False, False, True], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_13b3f24a6c3269475fbc90679e4a2f3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_494d4ee3688a3d2fb8a99bb607cfdaf3
        def get_inputs(self):
            return [
                paddle.to_tensor([False, True, False, False, False, False], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_d0e44ec36e55816d9a40120b44708842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8bf49f287ffe1f800204cd3227bd9079(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_2b70fff76ea2884eeffe289b4931a7b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_5d2933a24846aa8109b5063bc6d6f80e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 76], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf2cac5a341bf1cd301c6fd918670ea9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5d2933a24846aa8109b5063bc6d6f80e
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 76], dtype='int32'),
            ]


    class TestPrimitiveOp_701bc56b1a57ab0252dbeb38b7baad7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d148f8c73a8bd2efbb938d71864dd29d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_49f8e13d5fb43c52e0891fd03468c740(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8859282d91d626b01597bc1424cdc7a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49f8e13d5fb43c52e0891fd03468c740
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0dc4ed1a98895912c3335eb5f9b12dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_048b9a55842cbd3d44b0dfe9073d92d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49f8e13d5fb43c52e0891fd03468c740
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0dc4ed1a98895912c3335eb5f9b12dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1f3aa8b7105bd71844a95c1b689f72d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5912faab2e7304099c333c2273a02969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(28, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_455fb2e4dae3b77bba7a7e490c4db660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(50, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_327c632de48419ebf0a0cec6e920bbf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9f017c817c5e5e2528e72b5dca4c1d28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4116, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6af0567ec8e4263875543d74dd97d6b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_377052d9e5a4d1685477b66be850eb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_9b1ac2037b1e54761a9accca33288d97(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[80], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_116669c6651ed97024ce9c3ce82240cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9b1ac2037b1e54761a9accca33288d97
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[80], dtype='int64'),
            ]


    
    class PrimitiveOp_965c6159c94f14be58432318bbce7df6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[40], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ea9d6cf3537cad88e239f9c1049c1463(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_965c6159c94f14be58432318bbce7df6
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            ]


    
    class PrimitiveOp_167f2cda0ab3f54c118f0f2a29fac59e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[20], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a02b4d25c76f7acc18efe6ace66ee7f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_167f2cda0ab3f54c118f0f2a29fac59e
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='int64').reshape([20]),
            ]


    
    class PrimitiveOp_24740f8175dbc8b35cd2e36ae8654af8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[8400, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8b57c82f01cea8bfe4808d7f93c54371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24740f8175dbc8b35cd2e36ae8654af8
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b57c82f01cea8bfe4808d7f93c54371(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_24740f8175dbc8b35cd2e36ae8654af8
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dbb633fb9b34aa398a84d4fa3e947da0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f1bbdac3abf2e0ed7298655316668071(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbb633fb9b34aa398a84d4fa3e947da0
        def get_inputs(self):
            return [
                paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_85e365607087a27225ad319e4d9b251e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.18319472670555115, 0.21686814725399017, 0.4732165038585663, 0.03563392162322998, 0.26207268238067627, 0.130903959274292, 0.44992533326148987, 0.0970231369137764, 0.4192184805870056, 0.3830379247665405, 0.41160717606544495, 0.21541158854961395, 0.13209235668182373, 0.42042574286460876, 0.40241438150405884, 0.11916881799697876, 0.17227095365524292, 0.34461653232574463, 0.4960813820362091, 0.1515713632106781, 0.1289307177066803, 0.3049981892108917, 0.004009498283267021, 0.4341105818748474], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_607138d5e5c2c75968ff59b90b76c828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_4d18433a3aeebb4a7808bfe4ef0eb253(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            ]


    
    class PrimitiveOp_8f991e281c3c03a6205df9642056e585(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dd26d5811a5726eb57032ca21cc53278(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8f991e281c3c03a6205df9642056e585
        def get_inputs(self):
            return [
                paddle.to_tensor([0.8015309572219849], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_414d6fe46a236283c65b74a3b1087e4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_a9d178494b67885762dbad6e58b22c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_b422abe97bc37e08f8628e492710b4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a9a64187dcd81608972c80858e5ab866(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6069, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e8d86a60943179cec504ce946f9a53fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c1b2d05289668d2976c3b5433f90b3e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_858763e42bb3d80696a704489452e7cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d9b616bd86227cdfb68b5064b8488614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_2b92d1a4db78c9e443bdc18dc66e5a14(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4b436abe1d2b9e3036a642304ec891d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_d0e44ec36e55816d9a40120b44708842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d0e44ec36e55816d9a40120b44708842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_36e3683c0890e5971abe9ee54ad90b75(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3549], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c6542ab2a481d5b9b772f16d10ee7708(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_36e3683c0890e5971abe9ee54ad90b75
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_37ef1a1c17d55f485dd9bdc8f13dae7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbb633fb9b34aa398a84d4fa3e947da0
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 2], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c1d92c52dce8caa0157fff62ca008e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.05085877701640129, 0.22930851578712463, 0.13148881494998932, 0.16494080424308777], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_9c9d2141ba1a408b0a40625974c81ea1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_ebc55a7f92e53d48f1702130b3a1afd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c7db86aae363005495679f570fe11305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(52, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e6c5170415d39acd221746109a801283(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(202, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_9b4dac631b58be7f91574ed87c93bf49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_d72fa4a2e1855d2d10160ce0a7b68bb0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f69cd8e75896860ea4530970bd326d35(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d72fa4a2e1855d2d10160ce0a7b68bb0
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype='int64').reshape([14]),
            ]


    
    class PrimitiveOp_dbf61a007038b8a0033ad4d4b66b4017(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14, 14, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_14da1f411a322a97317a8816a6b299dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dbf61a007038b8a0033ad4d4b66b4017
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1c4ff367e07f67b2db49d6df626c4a2a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[14, 14, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a2d445e7e771b62a9dde632567b6e27d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1c4ff367e07f67b2db49d6df626c4a2a
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dd924bf7f4a06b0d09bb6397fbb19a1d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11b937dea2c05e7699f1cd3b0184932e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dd924bf7f4a06b0d09bb6397fbb19a1d
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype='int64').reshape([28]),
            ]


    
    class PrimitiveOp_b6a964fd5ba0d2fc04ae61aaa659985e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28, 28, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9c63b7baf85233442ee2e34d2fcaf0fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6a964fd5ba0d2fc04ae61aaa659985e
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_14994c8139b7ca1f9eb56df47ee856f1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[28, 28, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7012fe1726325a83502a5f3341408595(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_14994c8139b7ca1f9eb56df47ee856f1
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_83fa5d61de5353900422a866d46ebcfa(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f15e638425879eefae55d2eb45a586e5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_83fa5d61de5353900422a866d46ebcfa
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[56], dtype='int64'),
            ]


    
    class PrimitiveOp_97b19bc246ceab01f0adc92aa661b635(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 56, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1ae2faccd5347b0b00169d9cb5d1b42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_97b19bc246ceab01f0adc92aa661b635
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9ad539f51be05eaff98b1bfc44f50af(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[56, 56, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5ada9e471186e01160b359fd27b603e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9ad539f51be05eaff98b1bfc44f50af
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d3977de51310b5b675e68ae607bed020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d3977de51310b5b675e68ae607bed020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_74b3b39940594e7ed3f79b7c29704a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_74b3b39940594e7ed3f79b7c29704a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d456125c20e8709f51d345846588b638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d456125c20e8709f51d345846588b638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_df23cb82a6abc405e332e66341a12a53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4116], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11c7b04a2b9b36ac979ed38a539625de(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_df23cb82a6abc405e332e66341a12a53
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_0385ac7e85164e92078c5d32a9dc4ca7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f267e329be7ff70a344a679bc1ce1e53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0385ac7e85164e92078c5d32a9dc4ca7
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ad14def99aa6443e82e74ad7e4589280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0385ac7e85164e92078c5d32a9dc4ca7
        def get_inputs(self):
            return [
                paddle.to_tensor(7, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_dbd46434a3d4d6668eefa389a8c47a89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed57fc976b74c7264c41a2ae1d62b306
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_df07556bcb50eb62674a9d3071e563f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_327c632de48419ebf0a0cec6e920bbf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d456125c20e8709f51d345846588b638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_54fb64ca06dcaa8c177284545507c514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_889f2146b1bd0013685cc0f9cc4f6bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ac913dc8c390be08efafea79d58db171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_35a5c89972ddb942e7dfe01e933a1101(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_13ece5edb03445612131a3f883776261(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_2b77422da737b8003fb201157908d80c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(14, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_745ff236856c5cb42be33532c577101d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(25, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3468979542df31eae0b38de024b22b01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e9c0e35484a0c31c4eb7896fe3ab29bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_a26bde6a7079b5ac5070c9d61f4018f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6419d71a3338c9353b7f4e569afd786d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_319b4a8ed2984a8d71256678f7e8041d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_85cd68a8179368438e420b409b3bed26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_69dad57109fe91198732d3f72b08747b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc4f829001b27144f1e99a9169ab8ea9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_d186b0b1eb608d724df2a2e5776be72a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c43d3f1ea3667c3516ff454fed853da5
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ca76330bd662f8046df8d19ef298e45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_25378821ff4b4d5eafd4b25e944c118c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_e160310c4aab632808e3f2ca39d9a4ac(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_628303a6e63369fdd13791f3d56cc4cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_46d17ce95919333b32d23f2a3e47b6cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2f2ef3c58d0e7c3bd67559778c172373(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_9426794fb901f6786fc5d6dac78fc06b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(9261, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8e2b45b31a7cca0e516d18f74aafb297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_c10aa1b3eb1b5c7a9106a23efb24ef13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[68], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0398101b03e2464e660cdebb59399dcb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c10aa1b3eb1b5c7a9106a23efb24ef13
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[68], dtype='int64'),
            ]


    
    class PrimitiveOp_d5db9826bc22015ea6667a03cf17ec32(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[34], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2178ddf2c8ca35f13e355a3513bf00f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d5db9826bc22015ea6667a03cf17ec32
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[34], dtype='int64'),
            ]


    
    class PrimitiveOp_60ba451ba2ab1f6c6d1dd68739e6b8bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[17], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_421597c2f998c39ccf335b0bf754c068(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60ba451ba2ab1f6c6d1dd68739e6b8bc
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='int64').reshape([17]),
            ]


    
    class PrimitiveOp_efec2b60305d3b30f3bb83f59e686288(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6069, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7ac24d4bf403c7a9f2e0ea74abab308f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_efec2b60305d3b30f3bb83f59e686288
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7ac24d4bf403c7a9f2e0ea74abab308f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_efec2b60305d3b30f3bb83f59e686288
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0da1b6c314859a47b30fa02fd594f9cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e238e4e14133595de0679e12bfaca88c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c72eb712919297e6293b36658e40d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0da1b6c314859a47b30fa02fd594f9cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e238e4e14133595de0679e12bfaca88c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c72eb712919297e6293b36658e40d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0da1b6c314859a47b30fa02fd594f9cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e238e4e14133595de0679e12bfaca88c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c72eb712919297e6293b36658e40d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_11b6a9848c4c138530d8d4c0d7d4cf2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_2b68d8d45a6e39fb9664d50fb81373b7(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_68d7aa7ee4a656445cf7dbc5a4e0643b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b68d8d45a6e39fb9664d50fb81373b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33c9044d694621a45b562f141809bb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_b422abe97bc37e08f8628e492710b4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_a9d178494b67885762dbad6e58b22c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_414d6fe46a236283c65b74a3b1087e4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_05c90ab4b6b7ff85a11064a85196c209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_05c90ab4b6b7ff85a11064a85196c209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_255c096b7b364d588899754d821ee37c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([16, 16, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f533630cfb7c54f7001d881ec2bf0f93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f533630cfb7c54f7001d881ec2bf0f93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    class TestPrimitiveOp_1e53c19a550c6d6f2caa3e56ef99e42a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0a164a1ccf9ff1d8317575ba884922c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(2100, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a5ee99c7706e5dc7280391b75c69fb89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e238e4e14133595de0679e12bfaca88c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c72eb712919297e6293b36658e40d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a5ee99c7706e5dc7280391b75c69fb89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e238e4e14133595de0679e12bfaca88c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c72eb712919297e6293b36658e40d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a5ee99c7706e5dc7280391b75c69fb89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e238e4e14133595de0679e12bfaca88c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c72eb712919297e6293b36658e40d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_11b6a9848c4c138530d8d4c0d7d4cf2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cb93b48df5baaef8a24a0bffeddfa4ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b68d8d45a6e39fb9664d50fb81373b7
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_33c9044d694621a45b562f141809bb13(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9b4dac631b58be7f91574ed87c93bf49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_136718bc3f7a3a88102e2d2d0e6d0f49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c9afa7dd6cd33ab5914867f1fa05c38d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_adea939fedbfa3fa65e61dcf2eda6f6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_19956c39cd74a109674b43dadd417461(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_8e0491cecfffeff8ee27eb4a779bf0a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_770831129b6d9d0217e6cce876f33d45(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_1f3aa8b7105bd71844a95c1b689f72d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8104d63fd07c8899143578d45a0220c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_005908c97f6102e0967b3b945160b113(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_64779df6d4f679da7ea20c47088c4e8f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad821bcab19e7e103f8425314f0c1b26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_e73dbe1b0aaa564b0db8388c3fbf43fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b32672bbb6b68bc28f48c4a33b4c4c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_978e5ff61942cb6663ef820ab31e7b56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_54a7d28a5a435f5ca6d52c3a46c5a608(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_93dde0adb49f16dadb978d29345438da(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_37fdf0805fc672335e13063f4987bc10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_94cf134f7c8f54c820830ca72a13a8fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8902609794c66efe1477400c269d5657(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7020f85e47c4842fd33575860d3b2abd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed57fc976b74c7264c41a2ae1d62b306
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f0b1f59ff7885ac9dd8a7be6f0459c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11109, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2cb575041940ca0d3027de6baa885e5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c13bf70cb6888b454b1b04bfbb9799ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc4f829001b27144f1e99a9169ab8ea9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_759d5fc3a8d179744da8196ad6b8af1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a3c36447721b7b29c0a891d401339b
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_4616d5206574316a91bda8b13e625d86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a3c36447721b7b29c0a891d401339b
        def get_inputs(self):
            return [
                paddle.to_tensor([11], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_81730dc23c99baafe9765021dc33f2af(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a3c36447721b7b29c0a891d401339b
        def get_inputs(self):
            return [
                paddle.to_tensor([384], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_9f83ad6103cde9bf5651095e66733bc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a3c36447721b7b29c0a891d401339b
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_f71a23c93311c62812dc67b1c306c91c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e0a3c36447721b7b29c0a891d401339b
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
            ]


    
    class PrimitiveOp_2efae52772cdec14f6ef3da9c7b9d8ff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[152], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6123845522e00889f00a878a810dcb4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2efae52772cdec14f6ef3da9c7b9d8ff
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[152], dtype='int64'),
            ]


    
    class PrimitiveOp_87f66f8c0efea1107f1a7a3060a8d68b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_57ee8e2281519fd69d463a06597a6540(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_87f66f8c0efea1107f1a7a3060a8d68b
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[100], dtype='int64'),
            ]


    
    class PrimitiveOp_7cb1d42913fb902f7359c12e9abee61c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 152, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be91df3370fe0088a2e82a3f4f0aae62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7cb1d42913fb902f7359c12e9abee61c
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_46b4752ce136af016981920f06bafa29(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[100, 152, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_11cf0530f08ae87658b695061c178d85(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_46b4752ce136af016981920f06bafa29
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3c310a1f3db7d96be5f81b9a5450b34c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[76], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f6af547d841a01bb519b790915b06586(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3c310a1f3db7d96be5f81b9a5450b34c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[76], dtype='int64'),
            ]


    
    class PrimitiveOp_49c70186ebc755d28c8c295a9b47ca39(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_1e89909c56c589120a71237d6322aaf6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49c70186ebc755d28c8c295a9b47ca39
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[50], dtype='int64'),
            ]


    
    class PrimitiveOp_7eb67108abd1c2981f907359acfdee65(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50, 76, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f893af8a10886bb15a1a3914e7542078(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7eb67108abd1c2981f907359acfdee65
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c3f8f22666842f59648f4506f2ad56e5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[50, 76, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3166641e81d1ee860ec4c9940aa9f800(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c3f8f22666842f59648f4506f2ad56e5
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2b55e16c5780079ea0e67f1043b179f2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[38], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a4baa7804712675dda073be4c06caac1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b55e16c5780079ea0e67f1043b179f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[38], dtype='int64'),
            ]


    
    class PrimitiveOp_b7c8d0f6f17fd637c8ccc346dff01028(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d371f33ab182ecffad43fe46f03f8f6c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b7c8d0f6f17fd637c8ccc346dff01028
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype='int64').reshape([25]),
            ]


    
    class PrimitiveOp_5ca9d63b114787e02346f253e0976811(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_74a7bef31078c906254385eb580f86b7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5ca9d63b114787e02346f253e0976811
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_477c8fc3d136254434ae6082e89fc40d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[25, 38, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b99461f253ec0c0cf8764a44a67b0df4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_477c8fc3d136254434ae6082e89fc40d
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cdd5542cd161a1d228c57abfe38fd50d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[19], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b7dfb684c7fe41419de9a1e3b83755eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cdd5542cd161a1d228c57abfe38fd50d
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype='int64').reshape([19]),
            ]


    
    class PrimitiveOp_5453a082900c0666a80b3185c1b575b5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bc48a0e0c897a9b83b2358bea38888d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5453a082900c0666a80b3185c1b575b5
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64').reshape([13]),
            ]


    
    class PrimitiveOp_fdd2d7a8d875bdaf48e480d2f2368ddb(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13, 19, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_faf0ed1b2217e7e0824aa4e48cd1838d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdd2d7a8d875bdaf48e480d2f2368ddb
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7aedb4f2bc3e65eb303de7dffed772a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[13, 19, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a606cf61dfc137b20f73c8c6efabf8f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7aedb4f2bc3e65eb303de7dffed772a0
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_2d5840f1baaa708fc8fb415d3e624130(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[10], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6c9f16f4b1b10fe8439416f1251a7243(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2d5840f1baaa708fc8fb415d3e624130
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64').reshape([10]),
            ]


    
    class PrimitiveOp_f35adcdad9625771c21bf1022afca5ac(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41cccc9988fe4117740a15a822b5ed5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f35adcdad9625771c21bf1022afca5ac
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype='int64').reshape([7]),
            ]


    
    class PrimitiveOp_147c3d572dfdaeee8f128ba1e0cc4b79(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 10, 4], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1a469ff474a34a131b351afa329da63(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_147c3d572dfdaeee8f128ba1e0cc4b79
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cee43e38bc2aeb71c449c412288d4732(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[7, 10, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3745a61591480761bcc45a15354382f8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cee43e38bc2aeb71c449c412288d4732
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e5ba1d711cc9e33554f8b3bbb9f855bc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0e90a76585aac6c1cf87e8bd9a734f60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e5ba1d711cc9e33554f8b3bbb9f855bc
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_377052d9e5a4d1685477b66be850eb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cd1365feb77ef51a3eb0e7821dea5b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_49f8e13d5fb43c52e0891fd03468c740
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0dc4ed1a98895912c3335eb5f9b12dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6578f132bd93044befa764afcc2dcf61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.023670200258493423, 0.06822778284549713, 0.09304618835449219, 0.1754312664270401, 0.09071575105190277, 0.4499177932739258, 0.03357797861099243, 0.24033771455287933, 0.2845117151737213, 0.14653056859970093, 0.3833077549934387, 0.40311363339424133, 0.47563251852989197, 0.23145411908626556, 0.440609872341156, 0.4829781949520111, 0.3875403106212616, 0.4994688332080841, 0.3966350555419922, 0.34343641996383667], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_8264cd39f331fd263caf750a7328aed2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_d525a7ec543c638b0d932751beadc73f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_414d6fe46a236283c65b74a3b1087e4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_a9d178494b67885762dbad6e58b22c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_b422abe97bc37e08f8628e492710b4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_df07556bcb50eb62674a9d3071e563f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d456125c20e8709f51d345846588b638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_54fb64ca06dcaa8c177284545507c514(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_889f2146b1bd0013685cc0f9cc4f6bdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ac913dc8c390be08efafea79d58db171(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_d2fa1c5a0f519fda16f96b2bf9ba4caa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0993620e02ab4773a804e4938d2a387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_11d931fc36b98d51f2a65251a527dab0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e238e4e14133595de0679e12bfaca88c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c72eb712919297e6293b36658e40d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_21a2aeccbfe4956ec9eaf762d3001765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(3024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2fa0394b232a2d711df9c8cdf05f3908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    
    class PrimitiveOp_260183f550a6a3f30bf303324910b2b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[72], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7313db5d0bcc9e6bdadf78210da88a83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_260183f550a6a3f30bf303324910b2b9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[72], dtype='int64'),
            ]


    
    class PrimitiveOp_0cac4c629ac2039529b53e3bf4b005b0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[36], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_12ccd0790ae26de7653709dbc8474dfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0cac4c629ac2039529b53e3bf4b005b0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    
    class PrimitiveOp_7920e4b847859f535d3fec0c15900e51(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[18], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8ff0bf5d602fe39aaf8226420ce4e5fb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7920e4b847859f535d3fec0c15900e51
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64').reshape([18]),
            ]


    
    class PrimitiveOp_1a29159cbf5064d7656352e37c9c413f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6804, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_490e9eb5be1c4c7a787a110a5b92a11c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a29159cbf5064d7656352e37c9c413f
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_490e9eb5be1c4c7a787a110a5b92a11c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a29159cbf5064d7656352e37c9c413f
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_0dc0814351ec64f30d779f9caa160826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7479f9bf18797ca618a2b25a300fe26b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2c5e647946e4d928b1d92515dffeb614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a5ee99c7706e5dc7280391b75c69fb89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e238e4e14133595de0679e12bfaca88c
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c72eb712919297e6293b36658e40d56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_df07556bcb50eb62674a9d3071e563f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6af0567ec8e4263875543d74dd97d6b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_377052d9e5a4d1685477b66be850eb7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e88c391e76c460139833ca1dfcefcbfe
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cc01e720ba95a63c397ab2a8c6918a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_fb77a2a5f2543e9e193d4d3f8ca4e297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a91f1d3030b648195fd93cca47c43d57
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_c255aa1d8a045d7a35fe88d13f399080(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ae4e21426cf0ae46f0a14333905836ff
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_db05db8ef7b04ec6cb45bdd6a4ce4ecf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2362317998c47fffe52d05887d08c74c
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_d560259a2ef60b769b8d6474e67f73d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_393346219e5fd7167c8ed7e3b1aa3226
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a9c5db84266558077d8f1ff48bdb3fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_01b0f4176a77c4db88483fe96504a35f
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_0dc0814351ec64f30d779f9caa160826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_414d6fe46a236283c65b74a3b1087e4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_a9d178494b67885762dbad6e58b22c82(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_b422abe97bc37e08f8628e492710b4ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_68694b970158fd2dd9d6df5119b46364
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_c83a83a7af5f3080eefddf0971e5adb8(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_41d8756f875d4f1a8e3482ab01ffd1ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c83a83a7af5f3080eefddf0971e5adb8
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_49b01bf697fc4aead173e5def1358871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    
    class PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_410e0672d58ef41a9aa3413dfb08b023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_b20c838d27048c0f43f531b63b0ee002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_aabf46c0676de4f1c1f42c05b5cf74ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c85ee3e36c3aaaa5359c6026a40ed384(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc4f829001b27144f1e99a9169ab8ea9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_190fc76e757da9b94fe324eca693675e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c83a83a7af5f3080eefddf0971e5adb8
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ca76330bd662f8046df8d19ef298e45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ca76330bd662f8046df8d19ef298e45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_4adb8ddbd2c5d2cc72cecc77a6dcdffc(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f104370e5337510635b84a45d626b03c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4adb8ddbd2c5d2cc72cecc77a6dcdffc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100], dtype='int32'),
            ]


    class TestPrimitiveOp_add0663d391943550179c86fcff1e9e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([128], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_1149e9cd79456a3e36c1318884424a27(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([16], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7479f9bf18797ca618a2b25a300fe26b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c43475283c1538f0f01a8b16215da9bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[96], dtype='int64'),
            ]


    class TestPrimitiveOp_b71134913ab23a640cd6d780cca87f84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[48], dtype='int64'),
            ]


    class TestPrimitiveOp_35b70f80d14c7b90d5de3d5d31c6aab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64').reshape([24]),
            ]


    
    class PrimitiveOp_8aa8084c157bfb2233b7c78719313499(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c4a24dc21c18aeb68d19e8d8efb9f52c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c4a24dc21c18aeb68d19e8d8efb9f52c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_290755d51d43534f7521f66b2a924325(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c43d3f1ea3667c3516ff454fed853da5
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_c9c515a20a0210e99220dcd0c095e76d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d3977de51310b5b675e68ae607bed020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d3977de51310b5b675e68ae607bed020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cca141ea5a2d6760c7232b0f07517ec9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.37460729479789734, 0.4844105839729309, 0.181888610124588, 0.44207683205604553, 0.02534153312444687, 0.3211316466331482, 0.009519957937300205, 0.40302348136901855, 0.22982287406921387, 0.10425077378749847, 0.3179247975349426, 0.40923941135406494, 0.2358681559562683, 0.14795830845832825, 0.06801445782184601, 0.4440118670463562], dtype='float32').reshape([16]),
            ]


    class TestPrimitiveOp_5b22f73a91049458cc542f6eec4653df(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_5063f03bbc4e5647c34bec6cab92fadc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_f8a99e88f8e25d2ac557ce3e4611ea68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(7581, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_1f3aa8b7105bd71844a95c1b689f72d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_157dd9fa55e0486bab7f9bddba2c27a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4725, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_450d8a9dafa9e6e3084f554adb7cf93d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc4f829001b27144f1e99a9169ab8ea9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_950609f8452d9f93dc5e0b9905db03f4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(577, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2fa0394b232a2d711df9c8cdf05f3908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f1b119b493ee43bb642656577972ac2f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([2], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d0e44ec36e55816d9a40120b44708842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_fe6433791a798ca709fd81d432dbec0a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.bool)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf3961d572ab55e680061d39870c3a7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    
    class PrimitiveOp_7525a9241209bee6cf885f22cd46274c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cdb7fb9a2fa838d6b9b5a126165aee69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_717152d127843ae7219ede1b42024764(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 68], dtype='int32'),
            ]


    
    class PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4d6d3070c9763df0aab531a27d52c916(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([1827, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8721d3d89b9067f5189b90d19b54d03e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1827, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_48981f5deab36851a461f8586d0e4e20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_7f69b8259cd3844f5cceb36e6e24b709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8400, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_05c90ab4b6b7ff85a11064a85196c209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_bfaf89f1d1e5403cc54eb4a57116523b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bfaf89f1d1e5403cc54eb4a57116523b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_49b01bf697fc4aead173e5def1358871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(3549, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6e852d794f2f351c90a94db0cf6f8adb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9b3d15cd8e9b464b3531bdfb3843a6aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_06b32c084977cb4acd8c3dbe994c8c5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6799a8553fc5310ba815ad89bd2df28d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_0050711215b0434714c4991eb937a542(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7f041016a6275bf0c6e1925f7d62d35b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_ef982d754014da2e784c252a0f1d1d26(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([5514, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ed7e66372506117dc572f0fbaa8e6ba0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[5514, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3126f96e5643f74fd4f9e99ea447d129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc7448ad6283b9ad636ccc47187d247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8e2b45b31a7cca0e516d18f74aafb297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_efe176d135affaf53c74fccc1e39ff64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(2, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_be4563c484a7b87c1c0277fb3d46af0d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(98, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_fb7c70c6811d3b4fa6c85c198a8604b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(99, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_708a44f625e5732f7fb086b8ba1f1c38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.uniform([36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_980c19d67909f8c4d939dccc9ff26d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_980c19d67909f8c4d939dccc9ff26d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_818799804309fbf751b3924eeab0f59c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(192, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ab69522c3fcfef465391a82e73f15412(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dcdf4a5fe08c88bc8b2249f4e025969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 192, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8a77584d23b9d5477d6052959d54b0bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_1f3aa8b7105bd71844a95c1b689f72d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6e852d794f2f351c90a94db0cf6f8adb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b3d15cd8e9b464b3531bdfb3843a6aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 64, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8e2b45b31a7cca0e516d18f74aafb297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2569206b57261dea2d17cfe96bcb1e0f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_494d4ee3688a3d2fb8a99bb607cfdaf3
        def get_inputs(self):
            return [
                paddle.to_tensor([False, False, False, False, False, True], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_13b3f24a6c3269475fbc90679e4a2f3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_494d4ee3688a3d2fb8a99bb607cfdaf3
        def get_inputs(self):
            return [
                paddle.to_tensor([False, True, False, False, False, False], dtype='bool').reshape([6]),
            ]


    class TestPrimitiveOp_d0e44ec36e55816d9a40120b44708842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_cf3961d572ab55e680061d39870c3a7c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_cdb7fb9a2fa838d6b9b5a126165aee69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_88f4bcb5a1ae7d137759e610ad65a063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 76], dtype='int32'),
            ]


    class TestPrimitiveOp_14d354938b39940c303be1ae8f64a4b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([1799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_16ec812318f3d7d265ca332d7be86f3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cb9f62caa7e873a27b09a564c7d3c688(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ec0d5ce1c30f34171dee6232057546a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_4bc67156efc2b17ded2cbfe17024f0e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ec0d5ce1c30f34171dee6232057546a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1f3aa8b7105bd71844a95c1b689f72d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5912faab2e7304099c333c2273a02969(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(28, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_455fb2e4dae3b77bba7a7e490c4db660(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(50, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_327c632de48419ebf0a0cec6e920bbf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_9f017c817c5e5e2528e72b5dca4c1d28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4116, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6af0567ec8e4263875543d74dd97d6b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0847f50baa87c1f10300842a41c2c756(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8d824a2d37d228579f3345595a89bca3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[80], dtype='int64'),
            ]


    class TestPrimitiveOp_7b0f5846d663a075a14c321eb5028eea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[40], dtype='int64'),
            ]


    class TestPrimitiveOp_0b310d1bbc08dae6a97efb3d92babf03(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_be82c1e0b6bcfa10a82109bd01c26c31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_be82c1e0b6bcfa10a82109bd01c26c31(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([8400, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4a7625d5be3334a16efd9f3d19e13e2d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.int32)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ba661656a57bc48d669aafef45d5d387(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a7625d5be3334a16efd9f3d19e13e2d
        def get_inputs(self):
            return [
                paddle.to_tensor([128, 128], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_85e365607087a27225ad319e4d9b251e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.18319472670555115, 0.21686814725399017, 0.4732165038585663, 0.03563392162322998, 0.26207268238067627, 0.130903959274292, 0.44992533326148987, 0.0970231369137764, 0.4192184805870056, 0.3830379247665405, 0.41160717606544495, 0.21541158854961395, 0.13209235668182373, 0.42042574286460876, 0.40241438150405884, 0.11916881799697876, 0.17227095365524292, 0.34461653232574463, 0.4960813820362091, 0.1515713632106781, 0.1289307177066803, 0.3049981892108917, 0.004009498283267021, 0.4341105818748474], dtype='float32').reshape([24]),
            ]


    class TestPrimitiveOp_607138d5e5c2c75968ff59b90b76c828(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([24]),
            ]


    class TestPrimitiveOp_4d18433a3aeebb4a7808bfe4ef0eb253(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([24]),
            ]


    
    class PrimitiveOp_92e07ac4530c8acec740fc0b2c188572(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.cast(input_0, paddle.float64)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b3e898209c27a5ac8251b87bf1cd0611(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_92e07ac4530c8acec740fc0b2c188572
        def get_inputs(self):
            return [
                paddle.to_tensor([0.8015309572219849], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_410e0672d58ef41a9aa3413dfb08b023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_b20c838d27048c0f43f531b63b0ee002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_aabf46c0676de4f1c1f42c05b5cf74ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a9a64187dcd81608972c80858e5ab866(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6069, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e8d86a60943179cec504ce946f9a53fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5cd6df053a9e5c3af2084f16c27f1629(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_d0cf31baa7869c47484b3e6d267326e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_16866b2f2038d4b9aaf98cf0d471fe28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_a025518d685667f972938f6fd0260909(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([1503, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd9723df7d9bd2796ef8b321c8360327(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1503, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_d0e44ec36e55816d9a40120b44708842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d0e44ec36e55816d9a40120b44708842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_114761f2e2c2d76fc5eeb046976e65cd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4adb8ddbd2c5d2cc72cecc77a6dcdffc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549], dtype='int32'),
            ]


    class TestPrimitiveOp_16156281d744cb38131a18ddd6f928d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4a7625d5be3334a16efd9f3d19e13e2d
        def get_inputs(self):
            return [
                paddle.to_tensor([8, 2], dtype='int32').reshape([2]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c1d92c52dce8caa0157fff62ca008e55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.05085877701640129, 0.22930851578712463, 0.13148881494998932, 0.16494080424308777], dtype='float32').reshape([4]),
            ]


    class TestPrimitiveOp_9c9d2141ba1a408b0a40625974c81ea1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_ebc55a7f92e53d48f1702130b3a1afd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0], dtype='int64').reshape([4]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c7db86aae363005495679f570fe11305(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(52, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e6c5170415d39acd221746109a801283(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(202, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_9b4dac631b58be7f91574ed87c93bf49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_a5af0a693c5953028c29a37e5cf0e05d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype='int64').reshape([14]),
            ]


    class TestPrimitiveOp_44ebeac29022c8e0ae111e4feaed44d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b56b3d9bd392018050f53098228e9a04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([14, 14, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fc83a1375dce35be19b6d296cdccbfa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], dtype='int64').reshape([28]),
            ]


    class TestPrimitiveOp_56768b4e38832e8d31709229ead7aec8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0613d91ab0381652bde6e731bc2cc921(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([28, 28, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4d874f521fdaf954f71dde9919f90e89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[56], dtype='int64'),
            ]


    class TestPrimitiveOp_e9bbb520aebaa82694ba7f53a134a05f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2535093e2d00e668246ae547b3e7e83f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([56, 56, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d3977de51310b5b675e68ae607bed020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d3977de51310b5b675e68ae607bed020(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(13, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_74b3b39940594e7ed3f79b7c29704a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_74b3b39940594e7ed3f79b7c29704a92(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(104, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d456125c20e8709f51d345846588b638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d456125c20e8709f51d345846588b638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_37e664c2f2058c399a8577366882dcf1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4adb8ddbd2c5d2cc72cecc77a6dcdffc
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116], dtype='int32'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f267e329be7ff70a344a679bc1ce1e53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0385ac7e85164e92078c5d32a9dc4ca7
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_ad14def99aa6443e82e74ad7e4589280(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0385ac7e85164e92078c5d32a9dc4ca7
        def get_inputs(self):
            return [
                paddle.to_tensor(7, dtype='int64').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_41d8756f875d4f1a8e3482ab01ffd1ff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c83a83a7af5f3080eefddf0971e5adb8
        def get_inputs(self):
            return [
                paddle.to_tensor([300.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_df07556bcb50eb62674a9d3071e563f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_327c632de48419ebf0a0cec6e920bbf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_d456125c20e8709f51d345846588b638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_fada720e8cf0487ba5f241f2fa67f86a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_296ce42f13cc510807dfef5c6fde831a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_270fdae5bcf195f291b534e8a5807f4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_df02182f45456c97205b98558c9ca06c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([2077, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9bddfba3738fd2b4c543b9e9cb3f0aed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2077, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_2b77422da737b8003fb201157908d80c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(14, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_745ff236856c5cb42be33532c577101d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(25, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3468979542df31eae0b38de024b22b01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b9568b55404d0a86ade1ae03c79ca496(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_07da11294765ae161e8b2e60e6d85b49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_98ae56ed2c90c11b5c9cd47f8b7f3c3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_4ec590b2fe63bb96a502602760121a89(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([4628, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b61a3726f196f1afa652bab392342e8e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4628, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_69dad57109fe91198732d3f72b08747b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc4f829001b27144f1e99a9169ab8ea9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[6, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_d186b0b1eb608d724df2a2e5776be72a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c43d3f1ea3667c3516ff454fed853da5
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ca76330bd662f8046df8d19ef298e45d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_2bcef29fe9e8d011ce9dbd1f85c403b9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_e15ee16d7a48b537457b1a2e3fabbc8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6c6c46bc6566556dbe0e4feb0e121a95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_a2f67f784b52cad4f8f94937b88b1c49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([1101, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bad22f826c6a81159f0c86b040ac03ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1101, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_9426794fb901f6786fc5d6dac78fc06b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(9261, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8e2b45b31a7cca0e516d18f74aafb297(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(10, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_10def10b6a57107593b2ad5db0f2a474(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[68], dtype='int64'),
            ]


    class TestPrimitiveOp_4964d7333236d3d35a21eb10404446bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[34], dtype='int64'),
            ]


    class TestPrimitiveOp_04c94640d41aa6632654bf85e07124a9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='int64').reshape([17]),
            ]


    class TestPrimitiveOp_4408e44eff209b4a1bbe0d0a61826173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4408e44eff209b4a1bbe0d0a61826173(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3126f96e5643f74fd4f9e99ea447d129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc7448ad6283b9ad636ccc47187d247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3126f96e5643f74fd4f9e99ea447d129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc7448ad6283b9ad636ccc47187d247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3126f96e5643f74fd4f9e99ea447d129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc7448ad6283b9ad636ccc47187d247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_11b6a9848c4c138530d8d4c0d7d4cf2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7c4564d546b2ecd84ceacf06a91719f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80cde3d0cbf5f5b7bf0534ae628fcfd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_aabf46c0676de4f1c1f42c05b5cf74ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_b20c838d27048c0f43f531b63b0ee002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_410e0672d58ef41a9aa3413dfb08b023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_05c90ab4b6b7ff85a11064a85196c209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_05c90ab4b6b7ff85a11064a85196c209(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype='int64').reshape([16]),
            ]


    class TestPrimitiveOp_0fe5bc27af7749b8cc8e85bf12de66d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([16, 16, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f533630cfb7c54f7001d881ec2bf0f93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f533630cfb7c54f7001d881ec2bf0f93(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype='int64').reshape([8]),
            ]


    class TestPrimitiveOp_5e1ab84e4f14d5f7c60c4055b8c2a1bc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0a164a1ccf9ff1d8317575ba884922c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(2100, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_af554830f4a99595009ea5c221087c8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc7448ad6283b9ad636ccc47187d247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_af554830f4a99595009ea5c221087c8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc7448ad6283b9ad636ccc47187d247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_af554830f4a99595009ea5c221087c8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc7448ad6283b9ad636ccc47187d247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_11b6a9848c4c138530d8d4c0d7d4cf2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(2048, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5aceea875b7f572690c2d541c731f415(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_80cde3d0cbf5f5b7bf0534ae628fcfd0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2048, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_9b4dac631b58be7f91574ed87c93bf49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1025, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_136718bc3f7a3a88102e2d2d0e6d0f49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_02000fdb28bd4781394d0229215e9611(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_806c8eb06b40757e6c1ae17ab7c70348(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_94bc44541f3868abdc89a7c58092349d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_9b0751a88675deef28fe0d187a4bbdc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([2361, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e1e6137d28c91a5d6800ecaa5e417d8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2361, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_1f3aa8b7105bd71844a95c1b689f72d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(22, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8104d63fd07c8899143578d45a0220c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_3f38e9137c326e7ae6cf25c3e4a77400(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_0be9b2d4f6b38a707249357ab8b6de53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_180557534490d7c4958bf5a6e51c8223(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_7dddb23aca3138759ed4dd1c1442fb0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([3061, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce4bfcdc202a0e7ec6b41b8794a5d10f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3061, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_978e5ff61942cb6663ef820ab31e7b56(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_30865e2b9d2594ed62ad0477c6569bf3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_6a6bacb4827e539132bd879ed4d5624d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d64146b958ca45cd23d413c2507f0e01(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_691c796585ee828e0433e5bc8114ad66(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([3799, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd042373394fcd79f82e80b672ce4b96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[3799, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_190fc76e757da9b94fe324eca693675e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c83a83a7af5f3080eefddf0971e5adb8
        def get_inputs(self):
            return [
                paddle.to_tensor([100.0], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_f0b1f59ff7885ac9dd8a7be6f0459c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11109, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2cb575041940ca0d3027de6baa885e5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c13bf70cb6888b454b1b04bfbb9799ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bc4f829001b27144f1e99a9169ab8ea9
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2, 28, 28], dtype='int32'),
            ]


    class TestPrimitiveOp_dde9fc2b4b86db3dbf3e297e5f532918(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([4], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7da51c45ec7a85aca9487f1c62b55c00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([11], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_38d786f169dbda90cb1dcd3cc43f9c3f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([384], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_cf6b2e4b54b6858d691091566f8c2db5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([28], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_41524466c155b262dd184600cd3d9328(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([77], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_7c2dee3d6e3e65145f429585201f024e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[152], dtype='int64'),
            ]


    class TestPrimitiveOp_0f3d1d6e43e4d910ba6530458197a7c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[100], dtype='int64'),
            ]


    class TestPrimitiveOp_4db1956ca35bae533ad6d44056bddaae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6281fd2569aa09021b4921371e30b633(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([100, 152, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_193fdca319d782d1044d33b648a737e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[76], dtype='int64'),
            ]


    class TestPrimitiveOp_084b6499a92240a663f929722ba2a2c8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[50], dtype='int64'),
            ]


    class TestPrimitiveOp_33145ff61c572fbe805c40f923047b96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_23fc7758f862ecd4574f86e3bdf97fe8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([50, 76, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_568427e0a39bcfcbadbb7c54560a5418(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[38], dtype='int64'),
            ]


    class TestPrimitiveOp_0a3cb2ec42aee41e211736a8ecb1bec3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype='int64').reshape([25]),
            ]


    class TestPrimitiveOp_00dd47d434cdc909bdc72e7db82039bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e518b41fa3aec2e116cad4bd04760946(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([25, 38, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1c305e1f883211fa8f4998de8aba1019(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype='int64').reshape([19]),
            ]


    class TestPrimitiveOp_3380c144d42da2c2926e7a7d8d94f842(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype='int64').reshape([13]),
            ]


    class TestPrimitiveOp_db0029c992b1642ebc20d57b8838979d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_12c02a140e2b0e55a51bb4310a785c15(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([13, 19, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4f16aec7bad23b84f5251a917ec43118(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64').reshape([10]),
            ]


    class TestPrimitiveOp_6eaf5cd1710c86c1ed34a96a4ef2ec11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6], dtype='int64').reshape([7]),
            ]


    class TestPrimitiveOp_04d826a8b30aad0a43ab397a6cc504c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47ca557bd573eedb3f0d95341380facf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([7, 10, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_56bd4266ab133395a2dc7cf20110b735(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0847f50baa87c1f10300842a41c2c756(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b979bae06fc1f3f10964985032edffe7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4ec0d5ce1c30f34171dee6232057546a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 256, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d2ba5b8d1e99da4f58059fa2734004e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(6, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6578f132bd93044befa764afcc2dcf61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c865ae5640000cc53f2a53c8096377a0
        def get_inputs(self):
            return [
                paddle.to_tensor([0.023670200258493423, 0.06822778284549713, 0.09304618835449219, 0.1754312664270401, 0.09071575105190277, 0.4499177932739258, 0.03357797861099243, 0.24033771455287933, 0.2845117151737213, 0.14653056859970093, 0.3833077549934387, 0.40311363339424133, 0.47563251852989197, 0.23145411908626556, 0.440609872341156, 0.4829781949520111, 0.3875403106212616, 0.4994688332080841, 0.3966350555419922, 0.34343641996383667], dtype='float32').reshape([20]),
            ]


    class TestPrimitiveOp_8264cd39f331fd263caf750a7328aed2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_d525a7ec543c638b0d932751beadc73f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='int64').reshape([20]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_410e0672d58ef41a9aa3413dfb08b023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_b20c838d27048c0f43f531b63b0ee002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_aabf46c0676de4f1c1f42c05b5cf74ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_df07556bcb50eb62674a9d3071e563f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_02c898f992056ee1ebf940a47ce58d38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(43, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d456125c20e8709f51d345846588b638(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_fada720e8cf0487ba5f241f2fa67f86a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_296ce42f13cc510807dfef5c6fde831a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_270fdae5bcf195f291b534e8a5807f4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_498196620ed25652e94d63b267a80571(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([2088, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_69d812f54c4e35cca95a2dac1826d871(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[2088, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_59a59cb9a5d084dee8d73831e152327c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc7448ad6283b9ad636ccc47187d247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_d82423c8a71d8a5a0a3dc795d7ec53d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(256, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c8c821fad289aa731ca8c248f75ab383(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(8, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_21a2aeccbfe4956ec9eaf762d3001765(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(3024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2fa0394b232a2d711df9c8cdf05f3908(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([1], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_3d8fc2f578ecb1cc1264e738f3de53ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[72], dtype='int64'),
            ]


    class TestPrimitiveOp_980c19d67909f8c4d939dccc9ff26d33(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[36], dtype='int64'),
            ]


    class TestPrimitiveOp_bd91cac192c299366e415f2cfe1a7d58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], dtype='int64').reshape([18]),
            ]


    class TestPrimitiveOp_4dcd11879bb25e02b7e3b9a9a5f0bb6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4dcd11879bb25e02b7e3b9a9a5f0bb6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8aa8084c157bfb2233b7c78719313499
        def get_inputs(self):
            return [
                paddle.uniform([6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_0dc0814351ec64f30d779f9caa160826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7479f9bf18797ca618a2b25a300fe26b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_560d5c5253615d2d02c631e63e63da2f
        def get_inputs(self):
            return [
                paddle.to_tensor([8], dtype='int32').reshape([1]),
            ]


    class TestPrimitiveOp_2c5e647946e4d928b1d92515dffeb614(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([0], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_af554830f4a99595009ea5c221087c8a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dcc7448ad6283b9ad636ccc47187d247(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 512, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_df07556bcb50eb62674a9d3071e563f0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int64').reshape([1]),
            ]


    class TestPrimitiveOp_c0b01e5272cd4faccfca1c663ddb39f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_b60124f0172cdee3c7e488c6d6644632(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_6af0567ec8e4263875543d74dd97d6b4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6fa1c10deab44f4b67415ca6c5f38996
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0847f50baa87c1f10300842a41c2c756(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f55726e0fdf00fc0dc65dd047c8fee92
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 128, 1, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5f8231d25df43cb01681fb2c43fb7dd5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_cc01e720ba95a63c397ab2a8c6918a5e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8419fe993f498f2a363e70169ba064d4
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ef62174d9ae1010140dd9a1a3a3a74cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 4], dtype='int32'),
            ]


    class TestPrimitiveOp_b3eae0e356cea825a28ba23d1d7272bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525a9241209bee6cf885f22cd46274c
        def get_inputs(self):
            return [
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6dfb1788e06147be3fe7213967f766d3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fe6433791a798ca709fd81d432dbec0a
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 68], dtype='int32'),
            ]


    class TestPrimitiveOp_5b4e7b655ad2bfba1ddf952e55a11fa2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_eb5f0328af0d82fb9c72df0f5b73e7db
        def get_inputs(self):
            return [
                paddle.uniform([4270, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e762fba6a57c224e813c5edf67516813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d0ae5f7a833be4bef4f9818dfe0a6e6d
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[4270, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_0dc0814351ec64f30d779f9caa160826(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1174, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_312c792a28dc89ba048b57bbe1a587d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(512, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_0accbbb435d77d176a452d83d3852a1e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_27349785ed97dd518b2d11eb2e82bec0
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[256], dtype='int32'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_ae917e9a0140452273cbdbac780c92dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(32, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_aa8f5a669d820bb7895e5080b1c27197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[32], dtype='int64'),
            ]


    class TestPrimitiveOp_410e0672d58ef41a9aa3413dfb08b023(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([32, 32, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_86f0a9e1d4737af9c65ade845662398e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(1024, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_2faacb9d96a51c603ebbaa8ab9de6062(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(64, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_5cc0cd469add778d557f761ac5cabcae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[64], dtype='int64'),
            ]


    class TestPrimitiveOp_b20c838d27048c0f43f531b63b0ee002(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2307f38840c5d8c427d6a8603d33405d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(4096, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_3052eea361f2df7c91d01902bd692b3c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(128, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_47a546e9ddfd433896fbbf3e0dd3032f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7e040d4c6c12a1f3861e8bf4017ee5f2
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[128], dtype='int64'),
            ]


    class TestPrimitiveOp_aabf46c0676de4f1c1f42c05b5cf74ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa2e87e83df4e90223ae2de4ea6199a0
        def get_inputs(self):
            return [
                paddle.uniform([128, 128, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_367d08c891369ea5908a776934b6a0e9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(16384, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_7ac08799b7afb170250bf88d1a443639(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78af5d5b366e7dc2766db7a2f9c520ee
        def get_inputs(self):
            return [
                paddle.to_tensor(11, dtype='int32').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()