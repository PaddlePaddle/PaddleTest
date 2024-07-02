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
    class PrimitiveOp_f0dad5c21fa872adc4f200d758e99f78(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 500, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 500, 128], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_03a68d7f4ced91fbefd6061d011988a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0dad5c21fa872adc4f200d758e99f78
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_146c90df78eff245a901d95919972040(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 1], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_81fb2a8c7fde1410d8cf962c3832119c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_146c90df78eff245a901d95919972040
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_03a68d7f4ced91fbefd6061d011988a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0dad5c21fa872adc4f200d758e99f78
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 4], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ad5d45ae31afb8324ee0361bba222743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad5d45ae31afb8324ee0361bba222743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_2b864ef3a541a88139a19c55ad66749c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8a1bbaba4c5b37fb28f427935bd5a655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 68], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_87dfa0c28d5829bad632934da5a69fd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad5d45ae31afb8324ee0361bba222743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_43a4e820a6b8de58f4cac6e37f8b6867(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_43a4e820a6b8de58f4cac6e37f8b6867(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09bdecf876723b95e35d146946992813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_832719597831698058e41a3ba527dd21(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_43a4e820a6b8de58f4cac6e37f8b6867(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad5d45ae31afb8324ee0361bba222743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad5d45ae31afb8324ee0361bba222743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8a1bbaba4c5b37fb28f427935bd5a655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_da8d32427b8ddc7288c8fe00b82618df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, 76], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d7f77decb02ee48f7bda83ba66b39774(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_da8d32427b8ddc7288c8fe00b82618df
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ad5d45ae31afb8324ee0361bba222743(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_97fbd5a11cc57030ebfbc6cf7e6ed02a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_97fbd5a11cc57030ebfbc6cf7e6ed02a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_789c052b4c3b30914a8d664b5437c697(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_39e03f1267fc6ec5266aab9bf528dcad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_97fbd5a11cc57030ebfbc6cf7e6ed02a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd530c0393dd25f6098e275dcee95000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd530c0393dd25f6098e275dcee95000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_35238e1d506281aedc9adcd39e5fd9bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e1d813954c51e9f56a598df70461345a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd530c0393dd25f6098e275dcee95000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a2d973283ccc896200baa08d9678f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a2d973283ccc896200baa08d9678f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_22d99b1052bfde1eb4863cfb72836d6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e24bec76169e3ebaa9629b5e60153bce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a2d973283ccc896200baa08d9678f361(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    
    class PrimitiveOp_78038311630a91074690e15b7214bae0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle._C_ops.masked_select(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='bool'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f501b8560fbd640086de96944b60aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0f501b8560fbd640086de96944b60aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_5fc0dce34908fd209a72cfd5f530baec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_146c90df78eff245a901d95919972040
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_88097c3f92d544c80818b2977affdb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_88097c3f92d544c80818b2977affdb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_61701180c926803288444334032f9f68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_4baddc1d1c5d9df0137901a685e63b83(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_88097c3f92d544c80818b2977affdb62(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e06def29b2909b1631c43658ff3516ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e06def29b2909b1631c43658ff3516ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_27f1f658539fd92c983fe22d89aa6563(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_24030cc27a162269759f0977b90cf426(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e06def29b2909b1631c43658ff3516ab(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_46b9ee67c4ab8fd605d01c6b05e37616(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_46b9ee67c4ab8fd605d01c6b05e37616(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1a750c0b6588e3ee6fe6fa1dc33494d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d3edcd3dd0f7bd594e01423a36c9c410(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_46b9ee67c4ab8fd605d01c6b05e37616(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_29d9039fab97bcf403fe9db07013aaa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_29d9039fab97bcf403fe9db07013aaa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_de80b16a609dcc0e59eab31db52d28d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0da219b2cebf08dacae79ef198e121e2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_29d9039fab97bcf403fe9db07013aaa6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_69e5454ae9bd0f8a75a7b495d17ac2b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_69e5454ae9bd0f8a75a7b495d17ac2b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd530c0393dd25f6098e275dcee95000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd530c0393dd25f6098e275dcee95000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_35238e1d506281aedc9adcd39e5fd9bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e1d813954c51e9f56a598df70461345a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_dd530c0393dd25f6098e275dcee95000(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_03a68d7f4ced91fbefd6061d011988a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f0dad5c21fa872adc4f200d758e99f78
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1e6d47d0418773bb2f1b57c724cbe9f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1e6d47d0418773bb2f1b57c724cbe9f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a35919b7847a4e683365c5ae894efe1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_6ef85e13cf9a9fc86a497ef06b352172(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_ad6653fd506f42b3e3b95fda3ea33bbf
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1e6d47d0418773bb2f1b57c724cbe9f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f45c8d7fd1271811b45ad4bbea328337
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_818be3bd4d3ae38b7532aa96f4fac5bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_75340292fb5e14a8a65f91d1f4605096(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_818be3bd4d3ae38b7532aa96f4fac5bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_786b66d873654b2d8915a37c9023eab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_786b66d873654b2d8915a37c9023eab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8a1bbaba4c5b37fb28f427935bd5a655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_452cdd7b7775b39b9aecd8155ee24f29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_786b66d873654b2d8915a37c9023eab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aa812816f3ecfc743e1673935d0b32f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aa812816f3ecfc743e1673935d0b32f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_09bdecf876723b95e35d146946992813(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_77c064f559940606728826cc1c4b220f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_aa812816f3ecfc743e1673935d0b32f3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 11109, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 11109, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_786b66d873654b2d8915a37c9023eab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_786b66d873654b2d8915a37c9023eab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8a1bbaba4c5b37fb28f427935bd5a655(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_d32baaf27e15f9145601219921d3a68e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 76], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 76], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_786b66d873654b2d8915a37c9023eab9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3549, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f1b730b2a6c1096265a3f375a5d1eb46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f1b730b2a6c1096265a3f375a5d1eb46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_789c052b4c3b30914a8d664b5437c697(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_65c5eb90627813433a7d1b910156e9b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f1b730b2a6c1096265a3f375a5d1eb46(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 3024, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 3024, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_383a475023458892281f6bc394fc8c7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_383a475023458892281f6bc394fc8c7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_35238e1d506281aedc9adcd39e5fd9bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7a31d1d09ba86128d6e39d5a550ff37e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_383a475023458892281f6bc394fc8c7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_575e5c2b8073d7815ed9937538a07fd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_575e5c2b8073d7815ed9937538a07fd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_22d99b1052bfde1eb4863cfb72836d6d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_f3bb92a0f2f904abfb6830e34c60b16a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_575e5c2b8073d7815ed9937538a07fd3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 9261, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 9261, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0f501b8560fbd640086de96944b60aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_0f501b8560fbd640086de96944b60aa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_173bd0d3f061f5bb268be78b88304204(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2434, 1], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2434, 1], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_2b2bb78fc5250f99b6b2c2b428f9b464(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_2b2bb78fc5250f99b6b2c2b428f9b464(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_61701180c926803288444334032f9f68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_18cbbecd0835b8da9b73845891a9a218(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_2b2bb78fc5250f99b6b2c2b428f9b464(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 2100, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b8772835c9669998917d92a1ba69bb6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b8772835c9669998917d92a1ba69bb6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_27f1f658539fd92c983fe22d89aa6563(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1706bad55eefa2690b13e2d572679637(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_b8772835c9669998917d92a1ba69bb6f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4725, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4725, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8a192e43ad8e2b2357e4d476a27b2697(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8a192e43ad8e2b2357e4d476a27b2697(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_1a750c0b6588e3ee6fe6fa1dc33494d0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_9e43440b6487b6206454aaa7f8a64fdc(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_8a192e43ad8e2b2357e4d476a27b2697(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 6069, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_19abec07e19a53ee0865caab6ea5cbbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_19abec07e19a53ee0865caab6ea5cbbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_de80b16a609dcc0e59eab31db52d28d1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_ccc8fdcd3b23d6a64a9b6cab120ec39f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_19abec07e19a53ee0865caab6ea5cbbe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 7581, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 7581, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_69e5454ae9bd0f8a75a7b495d17ac2b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_69e5454ae9bd0f8a75a7b495d17ac2b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8732, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8732, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_383a475023458892281f6bc394fc8c7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_383a475023458892281f6bc394fc8c7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_35238e1d506281aedc9adcd39e5fd9bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_7a31d1d09ba86128d6e39d5a550ff37e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_383a475023458892281f6bc394fc8c7b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 4116, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_818be3bd4d3ae38b7532aa96f4fac5bd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 500, 128], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 500, 128], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_30ae81069c246cd519585efb63ea412e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_30ae81069c246cd519585efb63ea412e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_a35919b7847a4e683365c5ae894efe1a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2b864ef3a541a88139a19c55ad66749c
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_e8918050f583b2626db8cec544ec3548(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 68], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 68], dtype='int32'), 'bool'),
            ]


    class TestPrimitiveOp_30ae81069c246cd519585efb63ea412e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_78038311630a91074690e15b7214bae0
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 4], dtype='float32', min=0, max=0.5),
                paddle.cast(paddle.randint(low=0, high=2, shape=[1, 8400, 4], dtype='int32'), 'bool'),
            ]


    

if __name__ == '__main__':
    unittest.main()