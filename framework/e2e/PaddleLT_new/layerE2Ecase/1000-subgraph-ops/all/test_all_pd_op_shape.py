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
    class PrimitiveOp_60589a2ae4bcda323a148b184392c785(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 24, 36], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_029009bf40ee34431bebacc842284cd1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_60589a2ae4bcda323a148b184392c785
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d34f3c63eef870110ad222e6d867d6a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c06646636aab9728de87d0755ca1fad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_8a43ac399fd57c112d809272e57bae4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 128, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3d0e1b7550edbd6305bfdac0920fe37a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a43ac399fd57c112d809272e57bae4b
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_230ce9685b5bfc2d796d2c13ae7cf3a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3020aa007bac501c3783faad1d8148f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b77bff407f7d9af909fdeaa8edf373e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0667cb6d17e534178427bb04fbb72d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dc0f177b8f8a02d442028565ad15ae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2e31efe59f709a8f6f3de479a843afb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_17d08f145a9efed1b414724ee43e3f76(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_a6d5a8a01e742ee0856274503c0ba901(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17d08f145a9efed1b414724ee43e3f76
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6f708dd670b89593e996a1242f8a9042(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 320, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e1aea81eb18e3a237e59d8c31f852e54(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f708dd670b89593e996a1242f8a9042
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_aa231df342398b28ed68da6135524444(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8f4196acc03ab87a3bcd6f2b6b68d251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa231df342398b28ed68da6135524444
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_39beb773f712aa6ae90ae2906155e74c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 7, 7, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1476abfa29c5e352adfebe7be4aadb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39beb773f712aa6ae90ae2906155e74c
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1673222c17469d0ac3505e202ea19c22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a2b1ed812c4ca96b32bebb3d70a531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22ee7dd095f2d487202324735c754d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c06fbde5408dd7f6f10589f678747c9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92514695dcfcd10c003eb4ad2e9cc843(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9d818fee0f340aa47ce9510e6da6ed1c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 25, 38], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4e21a1594c77a5d9122ffd0407ffd2aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9d818fee0f340aa47ce9510e6da6ed1c
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_278ec4a349ed1ec562459f8dc8113e0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([12, 288, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_760943a41dd34e06ec791a0c278856f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6f708dd670b89593e996a1242f8a9042
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb571d0ea1d38ee7c2c821845f5b4802(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_392d74d18c27fcdf5183a67e635bee62(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 160, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3000f4c4aaecf3a79ca85558f69ab909(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392d74d18c27fcdf5183a67e635bee62
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c113f38c2e63670b29c55534c2adf74(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa231df342398b28ed68da6135524444
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24476a123ee1e353c658b47b82d43af4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17d08f145a9efed1b414724ee43e3f76
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_51e71d33f036556225664d7754cd353c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 14, 14, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af50f8626261ae59f6fc1c8dad973d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51e71d33f036556225664d7754cd353c
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fa0ab93e6e889b6636ed3045d840fc99(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 64, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_af7f55a9ffcdebc1fe0e2c6f71782465(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa0ab93e6e889b6636ed3045d840fc99
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5be55023868e3de7ffef232cf5df72e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_f95c27e38edf53b98b125c133347a8be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4068bb666cc654a6cdee441256adca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f95c27e38edf53b98b125c133347a8be
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_95c1d9bbceb9e578a560b94c2c32b3d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 28, 28, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_10d1556e957ee751f043c0f4d082158f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95c1d9bbceb9e578a560b94c2c32b3d1
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_de09c8ff9a8f8160f64a0b4e2c88208c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a43ac399fd57c112d809272e57bae4b
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15d817fa6632b9b67454d9082a568c32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_161f6f88f269f4518d492da8559d372a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_ed631d6502a1c84144eb5a9ecae068f4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 20, 30], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fe3948005b6ff11c1f7a196f0d2ba5f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ed631d6502a1c84144eb5a9ecae068f4
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34fa9ab7a1f3c33388ab9d9e3a3407d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6cf6d7c57da115d1abd7cd4ddfd1f96c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d0fd1d1f265c865c6ae61cdc68dd5a6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cf6d7c57da115d1abd7cd4ddfd1f96c
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_26d60417ddf49cc2079f94567fe0c8ab(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 56, 56, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6e91846033117113dcbc03267604325f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26d60417ddf49cc2079f94567fe0c8ab
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_093258fc646062dabbec8c434c359ca4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8416737bdd7a40a9febd815f1fb0ba68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_392d74d18c27fcdf5183a67e635bee62
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e362595c947674f213d27ee3a8f5f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a2b1ed812c4ca96b32bebb3d70a531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebf9b2e70175ab21d89ca88f1b7b2727(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e29add8e99598af57897c301164de874(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 768, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a8949e566a77452e7cbcf95bf50a06e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e29add8e99598af57897c301164de874
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_955a62b215bb3f6cad5ba18f444f934a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa231df342398b28ed68da6135524444
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_973c8507545a2ed0ae2e238a459f02f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_dec9df9653667bfccb7734b0be80b28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c74d5dc4a80df60fc328589b4d8c88d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a1ac4671c6079276c1b2d530cb6bc57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e48b9e0fc192f308ee81976e20343d53(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 192, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cc2617c0ec7cb97a5d9b62555e950078(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e48b9e0fc192f308ee81976e20343d53
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65aa70f79104ca60c36269353ab68d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa231df342398b28ed68da6135524444
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f09271132a057e1479fa69ad082387d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39beb773f712aa6ae90ae2906155e74c
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_1a5ceba44431469e395ed1669aa478a5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 3, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_73ef7d4f47cb8843976f10aeab49e9b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a5ceba44431469e395ed1669aa478a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4f1ba7071baf0cdc2a1b614fbd8a9b8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9f38cab4f59fe0ba3357becce7cedf22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f1ba7071baf0cdc2a1b614fbd8a9b8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4f304aaaa02fadd0aec85d130e1b86b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bb521b5f4278a13816c7c564dd0078a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a51a12edf1c23faabfd5de5b58898229(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbc2229853e4fe6708c90b57b8a41991(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_4dfbadbc8ee76b73f0f69047b4820c31(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_4dfd98e9f712338470b0fba18c0305dc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dfbadbc8ee76b73f0f69047b4820c31
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_973c8507545a2ed0ae2e238a459f02f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_522ab18a66ef4199832749cdc59fc4e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8e304281abbdf8cb48d1c3f1b53d945(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_acdf29b11d883bb8de2894f150b94500(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f99a22164b0faad08cc37c56348a0866(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acdf29b11d883bb8de2894f150b94500
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e26ddbcf039076604c429acaf4ccf11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acdf29b11d883bb8de2894f150b94500
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3fc688139700a74339f78c690101c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a4933dde5757d154974c08fdabda512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 28, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7af2fe138dec9433830eee492281f0a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a13a31f7d829f57814315e898dcbf58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a13a31f7d829f57814315e898dcbf58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a2b1ed812c4ca96b32bebb3d70a531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0fd1d1f265c865c6ae61cdc68dd5a6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cf6d7c57da115d1abd7cd4ddfd1f96c
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e91846033117113dcbc03267604325f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26d60417ddf49cc2079f94567fe0c8ab
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6764748547ae142c8f79d1a2d33b0607(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5871f768c8d58009c19a03bdf02ca181(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15855c5c863a19451a26482d3b61a856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b61ca7aade74dee11710af8bf8488ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acf841836633ed38464686f356f06c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f95c27e38edf53b98b125c133347a8be
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbbdaead9ba381b24f9412a90e1419f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95c1d9bbceb9e578a560b94c2c32b3d1
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fe7459cdaae5a512ba11d2203a5ec04f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17d08f145a9efed1b414724ee43e3f76
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5f03b022298e07297f31516191b3add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6323beca01c17399976b4ee1451fb24b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24476a123ee1e353c658b47b82d43af4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17d08f145a9efed1b414724ee43e3f76
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af50f8626261ae59f6fc1c8dad973d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51e71d33f036556225664d7754cd353c
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_49a8636b76aa7496f140d47050978f70(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4dfbadbc8ee76b73f0f69047b4820c31
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dfc7eb16d637876ff1842e4893020cea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fa0ab93e6e889b6636ed3045d840fc99
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2361e1d154810fc9b3bd0f0b22ffd7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 464, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e362595c947674f213d27ee3a8f5f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f9bda920a1ee5b4a52a562be232fb3a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1a5ceba44431469e395ed1669aa478a5
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fd3a2db7c3f13599b30ca44055287e50(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4f1ba7071baf0cdc2a1b614fbd8a9b8f
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a19c9867d171b604d7237e06528ac948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acdf29b11d883bb8de2894f150b94500
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b92b3a00063802a919af1cbc95b3fa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17d08f145a9efed1b414724ee43e3f76
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9e369b39d4f62f9f42a035ff1e01f04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51e71d33f036556225664d7754cd353c
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03a135ae037798f3c69a778c0a5aa405(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f9a4ab9bbb8eeaacc0e267abda5a37f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 14, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81c4c43d52977e6646626adc60ea5573(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cf6d7c57da115d1abd7cd4ddfd1f96c
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a36f2696adafbec9167ec47bc039935f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26d60417ddf49cc2079f94567fe0c8ab
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dffc9c8029a463673902adfc754d053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c98a57d3a767f9cd82b73cd8bef7aa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56aa3fb7fd6fbbc029212bad0e5dd57a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dec9df9653667bfccb7734b0be80b28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dec9df9653667bfccb7734b0be80b28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dec9df9653667bfccb7734b0be80b28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7ecf6f7fb0bbf0a3b565c7fb6cede3c4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 2048, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2384285def57a6a22e4a513216f53dc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ecf6f7fb0bbf0a3b565c7fb6cede3c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_acf841836633ed38464686f356f06c7e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f95c27e38edf53b98b125c133347a8be
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dbbdaead9ba381b24f9412a90e1419f6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95c1d9bbceb9e578a560b94c2c32b3d1
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8062bbfab73c03bc00e0f8acb4c2b7d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5680c7158ea3d870722c233be34951e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d97ebc16fafebc0f02bb7b01bac405e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a922e2f22fc6945feb8f8ecc3ff90f37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_535dc5617159f15939f272d1de086d29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7eabc4e98783fd04357e4e3527321e8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dec9df9653667bfccb7734b0be80b28a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5f03b022298e07297f31516191b3add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5f03b022298e07297f31516191b3add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5f03b022298e07297f31516191b3add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_66b31d0640275f0df8ea6155947a8ef5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7ecf6f7fb0bbf0a3b565c7fb6cede3c4
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b92b3a00063802a919af1cbc95b3fa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17d08f145a9efed1b414724ee43e3f76
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9e369b39d4f62f9f42a035ff1e01f04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51e71d33f036556225664d7754cd353c
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c9a4aa88b1c6035e974a852ce27594dd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 15, 25], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8d6e2c27eef2f9e7e8fd588b4e028d20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c9a4aa88b1c6035e974a852ce27594dd
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_24476a123ee1e353c658b47b82d43af4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17d08f145a9efed1b414724ee43e3f76
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af50f8626261ae59f6fc1c8dad973d79(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51e71d33f036556225664d7754cd353c
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_955a62b215bb3f6cad5ba18f444f934a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa231df342398b28ed68da6135524444
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcef3f48578577b065f9b7ce5896858f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e985f47dd4b71279a98e2e55cb93a66f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81c4c43d52977e6646626adc60ea5573(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cf6d7c57da115d1abd7cd4ddfd1f96c
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a36f2696adafbec9167ec47bc039935f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26d60417ddf49cc2079f94567fe0c8ab
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_547458bc82eee5dc79a8e02605662f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_37ff79e970f539ec03278bd5039cebd2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.shape(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 512], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_20ed19970f0d83542116690f46f147ef(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37ff79e970f539ec03278bd5039cebd2
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0e5a8b947bccb18666ef258b5950806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8f4196acc03ab87a3bcd6f2b6b68d251(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa231df342398b28ed68da6135524444
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c1476abfa29c5e352adfebe7be4aadb7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39beb773f712aa6ae90ae2906155e74c
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_65aa70f79104ca60c36269353ab68d96(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa231df342398b28ed68da6135524444
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f09271132a057e1479fa69ad082387d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_39beb773f712aa6ae90ae2906155e74c
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d37337e00866302baa3d9e4f553ae70c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_8a43ac399fd57c112d809272e57bae4b
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_775a3127a0e326cc0c155145511b27db(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acdf29b11d883bb8de2894f150b94500
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_512aa52c6986b53ecff496d9f8e8c8fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dc0f177b8f8a02d442028565ad15ae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2e31efe59f709a8f6f3de479a843afb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac838566bbaf88f374d5cd57e9d614a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d8b3c01d7fc2fa9304981fec57cd353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_210b3613a8ba559b4ac6ea5763a67596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_436e0843a56bb8e7dbcd8ba378ec51ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03a504b1545137fd6f3e6087581a6a38(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_37ff79e970f539ec03278bd5039cebd2
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d4068bb666cc654a6cdee441256adca2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f95c27e38edf53b98b125c133347a8be
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10d1556e957ee751f043c0f4d082158f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_95c1d9bbceb9e578a560b94c2c32b3d1
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb6af7b93e778cefc7bfce6709d16eb1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa231df342398b28ed68da6135524444
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cf2603dad5d1c0ccbd55dc66610846f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 232, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d0fd1d1f265c865c6ae61cdc68dd5a6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cf6d7c57da115d1abd7cd4ddfd1f96c
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6e91846033117113dcbc03267604325f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26d60417ddf49cc2079f94567fe0c8ab
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30c590228a0374380f6cedbfa272119f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_90277ac7ce6f6221bb84fd4160c0563d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9d3f1b82df82730e545f096a0fc1c7bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff55eee9eed6bf50524dcb00ea67fd55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e0e7c533579ca44e5f4abafa428ee29d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9b92b3a00063802a919af1cbc95b3fa5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17d08f145a9efed1b414724ee43e3f76
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9e369b39d4f62f9f42a035ff1e01f04(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_51e71d33f036556225664d7754cd353c
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c71e89073e0408bb2a21662755f6112e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d777682fb46b846b5800e7985f8b7ab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_17d08f145a9efed1b414724ee43e3f76
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b5f03b022298e07297f31516191b3add(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1540d19adaee6267b1bcabdef2a0740f
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36ab03d1c761170d115cff16d0fbd5c4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_acdf29b11d883bb8de2894f150b94500
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a13a31f7d829f57814315e898dcbf58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bca5ca2dc756f9b8f41d73cd8782e7bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 116, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3aba66c281db9a1a7cf75630f7a3176(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_aa231df342398b28ed68da6135524444
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_635cb64028359ab290c3b64cc6e1f62b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbbf4d8baaca95011ff1fc3e540e33d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e32cb7cd4aa509917978e2ba12723d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_81c4c43d52977e6646626adc60ea5573(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6cf6d7c57da115d1abd7cd4ddfd1f96c
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a36f2696adafbec9167ec47bc039935f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_26d60417ddf49cc2079f94567fe0c8ab
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c450055be38ee5f8cef9ad53e6f7f00(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 24, 36], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d34f3c63eef870110ad222e6d867d6a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8c06646636aab9728de87d0755ca1fad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a2fda8071473d0a9b96dd29f681d0e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_230ce9685b5bfc2d796d2c13ae7cf3a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a3020aa007bac501c3783faad1d8148f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 576, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b77bff407f7d9af909fdeaa8edf373e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 288, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0667cb6d17e534178427bb04fbb72d6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dc0f177b8f8a02d442028565ad15ae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2e31efe59f709a8f6f3de479a843afb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_63201faa2661baf88a5328a993cfa1aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_da6c06e35f35587c4028075f69dfa049(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([128, 320, 8, 4], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b309a2a6cd91ec9d99b1a45d1b95e8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10189cb95a383ac062f7fb6e41d510e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1673222c17469d0ac3505e202ea19c22(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a2b1ed812c4ca96b32bebb3d70a531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_22ee7dd095f2d487202324735c754d1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c06fbde5408dd7f6f10589f678747c9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92514695dcfcd10c003eb4ad2e9cc843(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([528, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_364906f365fbede518ac3eed1a9e5cc7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 25, 38], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_278ec4a349ed1ec562459f8dc8113e0e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([12, 288, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5384e58779e38298fae812f7a1232748(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([8, 320, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bb571d0ea1d38ee7c2c821845f5b4802(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47108e9260121e45e02b6ce478ad0a90(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0403e8519164cf3764d513266ebc3451(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 577, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0171778a844df5da5f399e5ceb7438f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2f5c185041f7fe406e5b601e6ecc948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b70d33f40e6cfc5d4b2f6d0cb51bbeee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([64, 64, 32, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5be55023868e3de7ffef232cf5df72e3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([384, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_571c0ff4c352a69ba236939fb2ccd197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9ab3250f3c5034e377ec65c4988e83e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8741bab97250a2ff06e5adbd7423f911(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([16, 128, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15d817fa6632b9b67454d9082a568c32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_161f6f88f269f4518d492da8559d372a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 112, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9115d16ff266f415af6865cfdfdb596a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 20, 30], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_34fa9ab7a1f3c33388ab9d9e3a3407d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a6290f6d828561a92bb2599f3adb31e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f6a62857ed4433623554d8a5b447180(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_093258fc646062dabbec8c434c359ca4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dda76d7f88ebfbb4c8fc483760137f57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([8, 160, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e362595c947674f213d27ee3a8f5f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a2b1ed812c4ca96b32bebb3d70a531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ebf9b2e70175ab21d89ca88f1b7b2727(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_635cb64028359ab290c3b64cc6e1f62b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eea36294064697ee74423bb702cc5c11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_973c8507545a2ed0ae2e238a459f02f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_547458bc82eee5dc79a8e02605662f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c74d5dc4a80df60fc328589b4d8c88d9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 192, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6a1ac4671c6079276c1b2d530cb6bc57(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([20, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_210b3613a8ba559b4ac6ea5763a67596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c7f92980809ed8af555e7ab2649de5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b27964e521b9dfd98b8f2bbc5bda7883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b617131a0f00f6ebc693ce4e20c19c9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 1024, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3acc375a3367e7f3cb155a92d95abfda(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 256, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f4f304aaaa02fadd0aec85d130e1b86b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([576, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bb521b5f4278a13816c7c564dd0078a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([22, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a51a12edf1c23faabfd5de5b58898229(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([96, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbc2229853e4fe6708c90b57b8a41991(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([12, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6c190271ff6e76f35e82422eed54deb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_973c8507545a2ed0ae2e238a459f02f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 64, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_522ab18a66ef4199832749cdc59fc4e1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([6, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a8e304281abbdf8cb48d1c3f1b53d945(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([960, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_316db396f9edb3441cc481dcda84597c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a87e304493da7844b85bfce0c19884ad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 256, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e3fc688139700a74339f78c690101c1d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([2112, 2, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4a4933dde5757d154974c08fdabda512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 72, 28, 50], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7af2fe138dec9433830eee492281f0a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a13a31f7d829f57814315e898dcbf58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a13a31f7d829f57814315e898dcbf58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a2b1ed812c4ca96b32bebb3d70a531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a6290f6d828561a92bb2599f3adb31e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f6a62857ed4433623554d8a5b447180(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6764748547ae142c8f79d1a2d33b0607(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 960, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5871f768c8d58009c19a03bdf02ca181(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 480, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15855c5c863a19451a26482d3b61a856(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 240, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b61ca7aade74dee11710af8bf8488ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd90cd524a982126620c09b87b5b9a9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdf09979b3b632f3c04557813768aab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbb67f3ac71ccda7d391bb687ee8fcfc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff55eee9eed6bf50524dcb00ea67fd55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6323beca01c17399976b4ee1451fb24b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 24, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0171778a844df5da5f399e5ceb7438f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2f5c185041f7fe406e5b601e6ecc948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8ba3145586069f7aff424af6c2df39ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b67f04323ca6b42199d11a6a2b35a373(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([16, 64, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a2361e1d154810fc9b3bd0f0b22ffd7a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 464, 16, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e362595c947674f213d27ee3a8f5f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_57a2b1ed812c4ca96b32bebb3d70a531(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 3, 512, 1024], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4e362595c947674f213d27ee3a8f5f9e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 32, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_db82c3ec6e9611dfc7812d8d7471ecd2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([8, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0776bfe4196b636b8f743c62147027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b1422c2cd0a5e0373fa14ddcce101fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_03a135ae037798f3c69a778c0a5aa405(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([8, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f9a4ab9bbb8eeaacc0e267abda5a37f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 144, 14, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_236c60a279a07aa72934cfe047922bb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d4a482c1bfecba20a90155b8f48391d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9dffc9c8029a463673902adfc754d053(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2c98a57d3a767f9cd82b73cd8bef7aa7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([240, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_56aa3fb7fd6fbbc029212bad0e5dd57a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([4, 32, 144, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_547458bc82eee5dc79a8e02605662f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_547458bc82eee5dc79a8e02605662f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_547458bc82eee5dc79a8e02605662f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dd426dc99b90999082aaaa17417de6a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cd90cd524a982126620c09b87b5b9a9c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bdf09979b3b632f3c04557813768aab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8062bbfab73c03bc00e0f8acb4c2b7d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5680c7158ea3d870722c233be34951e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d97ebc16fafebc0f02bb7b01bac405e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a922e2f22fc6945feb8f8ecc3ff90f37(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 16, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_535dc5617159f15939f272d1de086d29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 8, 8], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7eabc4e98783fd04357e4e3527321e8b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_547458bc82eee5dc79a8e02605662f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff55eee9eed6bf50524dcb00ea67fd55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff55eee9eed6bf50524dcb00ea67fd55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff55eee9eed6bf50524dcb00ea67fd55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_50bc75f95db037faf26c9acc27d5035f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 2048, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0776bfe4196b636b8f743c62147027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b1422c2cd0a5e0373fa14ddcce101fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_67ca922f7695b05708a0ef74912cbff4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 15, 25], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0171778a844df5da5f399e5ceb7438f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2f5c185041f7fe406e5b601e6ecc948(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_eea36294064697ee74423bb702cc5c11(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 1025, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bcef3f48578577b065f9b7ce5896858f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([44, 8, 288, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e985f47dd4b71279a98e2e55cb93a66f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 80, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_236c60a279a07aa72934cfe047922bb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d4a482c1bfecba20a90155b8f48391d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_547458bc82eee5dc79a8e02605662f6e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ccf16829bf40b73468f646cba3f5c3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a0e5a8b947bccb18666ef258b5950806(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5b309a2a6cd91ec9d99b1a45d1b95e8c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_10189cb95a383ac062f7fb6e41d510e4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5c7f92980809ed8af555e7ab2649de5d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 49, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b27964e521b9dfd98b8f2bbc5bda7883(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1ea424cd5819732f6166ab5cc0120f64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5680c7158ea3d870722c233be34951e8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_512aa52c6986b53ecff496d9f8e8c8fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([144, 4, 96, 24], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7dc0f177b8f8a02d442028565ad15ae8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 40, 128, 256], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e2e31efe59f709a8f6f3de479a843afb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 80, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ac838566bbaf88f374d5cd57e9d614a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 160, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d8b3c01d7fc2fa9304981fec57cd353(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_210b3613a8ba559b4ac6ea5763a67596(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_436e0843a56bb8e7dbcd8ba378ec51ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 96, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_621309f74d898d5e39ae96f308e781a4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 512], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_571c0ff4c352a69ba236939fb2ccd197(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 784, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d9ab3250f3c5034e377ec65c4988e83e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bf5de0f1506bf798db34a6813037970a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 1024, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3cf2603dad5d1c0ccbd55dc66610846f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 232, 32, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8a6290f6d828561a92bb2599f3adb31e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([43, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5f6a62857ed4433623554d8a5b447180(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_30c590228a0374380f6cedbfa272119f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([43, 40, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f7c69b80df12c057a96cea55675abe07(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 97, 97], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c47798edc4d1a73201f916b710450a9a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 8, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff55eee9eed6bf50524dcb00ea67fd55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_049fe3eb2f8cefef8b8fef1dd2862129(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([4, 512, 4, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e0776bfe4196b636b8f743c62147027(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 196, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b1422c2cd0a5e0373fa14ddcce101fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c71e89073e0408bb2a21662755f6112e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbb4c8d5196a216929275f0112eebf72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ff55eee9eed6bf50524dcb00ea67fd55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d750c5b074402fcfb93f410abb846012(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([4, 256, 8, 16], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5a13a31f7d829f57814315e898dcbf58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 128, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bca5ca2dc756f9b8f41d73cd8782e7bf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 116, 64, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_91c4f024fa746d7363277b501d5eb44f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([1, 1174, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_635cb64028359ab290c3b64cc6e1f62b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 768, 32, 32], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_bbbbf4d8baaca95011ff1fc3e540e33d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 384, 64, 64], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_3e32cb7cd4aa509917978e2ba12723d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([1, 192, 128, 128], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_236c60a279a07aa72934cfe047922bb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_194cb886dcbb7299a6667f3b5fbbe033
        def get_inputs(self):
            return [
                paddle.uniform([11, 3136, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0d4a482c1bfecba20a90155b8f48391d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0637c6e2eab689fbd97a63bcc2630a07
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()