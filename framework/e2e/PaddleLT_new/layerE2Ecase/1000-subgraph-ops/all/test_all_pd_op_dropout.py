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
    class PrimitiveOp_8a739d89fea50b4e6ea986ecdd63ec5e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.5
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'downgrade_in_infer', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3db48fe2e86d034315301c42f96918c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a739d89fea50b4e6ea986ecdd63ec5e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5217bb1bcae2e4727ce25cf1f7222d91(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.10000000149011612
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[11, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_38fab33529733ba518bef7dc482a888b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5217bb1bcae2e4727ce25cf1f7222d91
        def get_inputs(self):
            return [
                paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_3dbc2ce1f6dfff39ae2750810f5059bf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.10000000149011612
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd192636ea2c19e5af42f23ac9d74212(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dbc2ce1f6dfff39ae2750810f5059bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_244ae4160a15aa42fe444d124c9b1a9e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.10000000149011612
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 2048], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_478229a6042c34662829fbc6e5bb9fa3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_244ae4160a15aa42fe444d124c9b1a9e
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9c21916e8e2eef452c7da4c5c031eb1d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.10000000149011612
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1024], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_63fd70ebb67f85675b37a65f1d1329a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9c21916e8e2eef452c7da4c5c031eb1d
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c8202f8b5a8f69a9cf5443e128e1031d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.5
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'downgrade_in_infer', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_913273f236c33e02aab622a2225f02e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8202f8b5a8f69a9cf5443e128e1031d
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_cbd5e33e1b69d794239d1477290fcd0b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.20000000298023224
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e5cf1a9085fa7246bf54a1b4a61c39b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbd5e33e1b69d794239d1477290fcd0b
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_dc5c7a355ba01aa93872ccbe9cd50edf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.5
            return paddle._C_ops.dropout(input_0, None, input_2, True, 'downgrade_in_infer', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_be48a22418755eb9f30d52b972250717(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc5c7a355ba01aa93872ccbe9cd50edf
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5c0458acc40cf14473f2fed0f06ac615(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.20000000298023224
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'downgrade_in_infer', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_501df60caceb128e0dcd855e876c5883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c0458acc40cf14473f2fed0f06ac615
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c478b04d7f93cc4b68a1c1cfbd95bb11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c0458acc40cf14473f2fed0f06ac615
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_949ec43433a02f3a75d6ba12c7aa11b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c8202f8b5a8f69a9cf5443e128e1031d
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b04319f510e3cd31af6e3cd3cb743924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a739d89fea50b4e6ea986ecdd63ec5e
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fbe4588a26b6a7d996ff521f9c81f84b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.10000000149011612
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[43, 320], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_942b58c01ef3c0f9ead6a8d22c99e3c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fbe4588a26b6a7d996ff521f9c81f84b
        def get_inputs(self):
            return [
                paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8ddf98c4886e0717591be222ec1a17ce(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.05000000074505806
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_01742efc9c41382533e1619004238045(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ddf98c4886e0717591be222ec1a17ce
        def get_inputs(self):
            return [
                paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_32d6c4eb2039886e1f0db998b543dcb9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.20000000298023224
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3840, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cf7137065d43cbcfa9facb1986c8017a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32d6c4eb2039886e1f0db998b543dcb9
        def get_inputs(self):
            return [
                paddle.uniform([22, 3840, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f019cc9fe8af229c4fe6023e5567bf86(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_dc5c7a355ba01aa93872ccbe9cd50edf
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22394a2c298ff81e8da17603910299c1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_cbd5e33e1b69d794239d1477290fcd0b
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9fe1e6a04117e50832790902130b1e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ddf98c4886e0717591be222ec1a17ce
        def get_inputs(self):
            return [
                paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a276510567d7ae4c00a01777da311cd6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_32d6c4eb2039886e1f0db998b543dcb9
        def get_inputs(self):
            return [
                paddle.uniform([10, 3840, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3db48fe2e86d034315301c42f96918c6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a739d89fea50b4e6ea986ecdd63ec5e
        def get_inputs(self):
            return [
                paddle.uniform([22, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fb28b29ca050635c9909962c9862728d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.10000000149011612
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5f9efbd91766e014735c0ebc2550ae1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb28b29ca050635c9909962c9862728d
        def get_inputs(self):
            return [
                paddle.uniform([11, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd192636ea2c19e5af42f23ac9d74212(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dbc2ce1f6dfff39ae2750810f5059bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_71dcfef2a37b89eb33ef5ad03b736568(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dbc2ce1f6dfff39ae2750810f5059bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 2048], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bd192636ea2c19e5af42f23ac9d74212(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3dbc2ce1f6dfff39ae2750810f5059bf
        def get_inputs(self):
            return [
                paddle.uniform([1, 169, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6b9b005d0b9873b7147914ab10e636a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a739d89fea50b4e6ea986ecdd63ec5e
        def get_inputs(self):
            return [
                paddle.uniform([43, 512, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9bd3f853489709b9e23f636caa88127(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.20000000298023224
            return paddle._C_ops.dropout(input_0, None, input_2, False, 'upscale_in_train', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_aeac899436d6a39d78fdec0d8bdf66ed(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9bd3f853489709b9e23f636caa88127
        def get_inputs(self):
            return [
                paddle.uniform([43, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c620c2207ba41b8e01a54e019186008c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = None
            input_2 = 0.5
            return paddle._C_ops.dropout(input_0, None, input_2, True, 'downgrade_in_infer', 0, False), None

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7470d62d4dbad91a0685b2250f154dcc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c620c2207ba41b8e01a54e019186008c
        def get_inputs(self):
            return [
                paddle.uniform([10, 512, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_501df60caceb128e0dcd855e876c5883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c0458acc40cf14473f2fed0f06ac615
        def get_inputs(self):
            return [
                paddle.uniform([22, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c478b04d7f93cc4b68a1c1cfbd95bb11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5c0458acc40cf14473f2fed0f06ac615
        def get_inputs(self):
            return [
                paddle.uniform([10, 1536], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_19afb972a2c01e86df75953a90fcfc99(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a739d89fea50b4e6ea986ecdd63ec5e
        def get_inputs(self):
            return [
                paddle.uniform([11, 512, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b04319f510e3cd31af6e3cd3cb743924(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a739d89fea50b4e6ea986ecdd63ec5e
        def get_inputs(self):
            return [
                paddle.uniform([10, 2048, 10, 10], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5344c5c2a0e1c61074a9e43e33c270a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fb28b29ca050635c9909962c9862728d
        def get_inputs(self):
            return [
                paddle.uniform([43, 320], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01742efc9c41382533e1619004238045(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ddf98c4886e0717591be222ec1a17ce
        def get_inputs(self):
            return [
                paddle.uniform([11, 704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4704164a2a13bd8fb5563e48fe2d720b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9bd3f853489709b9e23f636caa88127
        def get_inputs(self):
            return [
                paddle.uniform([22, 3840, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d0bfc9903fe802e6c961bc38a5a9b4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c620c2207ba41b8e01a54e019186008c
        def get_inputs(self):
            return [
                paddle.uniform([22, 512, 12, 12], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1d4b406f8b887c50a042c0c7248a2a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9bd3f853489709b9e23f636caa88127
        def get_inputs(self):
            return [
                paddle.uniform([11, 1280, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c9fe1e6a04117e50832790902130b1e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8ddf98c4886e0717591be222ec1a17ce
        def get_inputs(self):
            return [
                paddle.uniform([43, 704], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66c40767559ef44175899633f39e5baf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9bd3f853489709b9e23f636caa88127
        def get_inputs(self):
            return [
                paddle.uniform([10, 3840, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()