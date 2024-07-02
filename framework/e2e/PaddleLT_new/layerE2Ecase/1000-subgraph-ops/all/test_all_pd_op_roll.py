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
    class PrimitiveOp_7baf538eea98c100223155d7a034739d(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 7, 7, 768], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b9970483501fc168cfa755fc50b22504(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7baf538eea98c100223155d7a034739d
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_696dc9d827f6d7cf07c1d1340a9b67be(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 14, 14, 384], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e6b3a64f68d86626e48d3ff074940512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696dc9d827f6d7cf07c1d1340a9b67be
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_9fed74d6e6d96ddb3f6495e566095d95(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7baf538eea98c100223155d7a034739d
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_bfde3bc3ca3c08f6a95a8f489e324d8f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 56, 56, 96], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_3bf34ef2298a181b38952803ae2bf62e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfde3bc3ca3c08f6a95a8f489e324d8f
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7525bf076fd2900e17c383dc4c550eed(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 28, 28, 192], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6fe75b7950d84fdf477bb02ba308b9d5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525bf076fd2900e17c383dc4c550eed
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e6b3a64f68d86626e48d3ff074940512(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696dc9d827f6d7cf07c1d1340a9b67be
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43e4efbafe7a1b26a802774efa2ed4d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696dc9d827f6d7cf07c1d1340a9b67be
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c5ed2cbb941f16619232382ff681909(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_bfde3bc3ca3c08f6a95a8f489e324d8f
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1daab48a81c32bed2fee329848a462a3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7525bf076fd2900e17c383dc4c550eed
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_43e4efbafe7a1b26a802774efa2ed4d2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_696dc9d827f6d7cf07c1d1340a9b67be
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [-3, -3]
            return paddle._C_ops.roll(input_0, input_1, [1, 2])

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bb1a22a39ef75a8b312cb7969d306821(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([43, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efd54dfb5c9e038a15aca3e1668b0892(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6eb1a36fd31924f6794a3ed421c63233(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([11, 7, 7, 768], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_af3ceb92edaeffac098fdbd8774a8c72(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([43, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6fc9b56840ebcdbbca3ec6203bf35536(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_efd54dfb5c9e038a15aca3e1668b0892(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([43, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6ac72b5d2a61e218f110a4269a27a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e9c51a3ba5ef3cd484d61bbef816a95e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([11, 56, 56, 96], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_31943a1fbfab7286e818119829eff6a8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 28, 192], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c6ac72b5d2a61e218f110a4269a27a10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e6f9a537803c81fe16c4ed96e4927355
        def get_inputs(self):
            return [
                paddle.uniform([11, 14, 14, 384], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()