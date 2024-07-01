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
    class PrimitiveOp_6d288336a9f8efdd256e8e3d480df475(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 28, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c78599a1a5f94f7120ebebcb0bd6befd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d288336a9f8efdd256e8e3d480df475
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_640005980e98bc23152f5e80700fb688(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 240, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_b89c21ddf4301e10abdf3729da1e6662(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_640005980e98bc23152f5e80700fb688
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_09e65187b8cb4febec5e642b51e28075(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 10, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_15ed3a5801da0d202c9d3f0b0cddb5b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09e65187b8cb4febec5e642b51e28075
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_fd6cd227eaad1d7d867bb2b41cab8bda(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 672, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2a675aec57eabee540cda02784fa6148(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6cd227eaad1d7d867bb2b41cab8bda
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c78599a1a5f94f7120ebebcb0bd6befd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d288336a9f8efdd256e8e3d480df475
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b74c4a1cb203e04e804e9f293db5d709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_640005980e98bc23152f5e80700fb688
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15ed3a5801da0d202c9d3f0b0cddb5b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09e65187b8cb4febec5e642b51e28075
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_22c2c0b908f3c98bbc02b4ba02fdc834(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 1152, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_47c04b649d4460274c08e877f637b02e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22c2c0b908f3c98bbc02b4ba02fdc834
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_db4ab4381ed71c213379c921f858216b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 48, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d5bca7f32e8c4144f786a503dad1d61a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db4ab4381ed71c213379c921f858216b
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a61b4caea3a11bb5607feb68ee34a6fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6cd227eaad1d7d867bb2b41cab8bda
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c78599a1a5f94f7120ebebcb0bd6befd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d288336a9f8efdd256e8e3d480df475
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_85343e1f2f9bbd0c9e3f706c66e0c39c(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 96, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e458cb1b8e035843309e0ec65dbb3312(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85343e1f2f9bbd0c9e3f706c66e0c39c
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_59a69d4a5cc207d2b595686163be37d5(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 4, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_fa289e8152778a0115a3c423d1e2256b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59a69d4a5cc207d2b595686163be37d5
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c2349e6ffe1603b8dc60fac7c33c18ae(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 480, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_89ee268719c2eb1274c3fe51b4f00562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2349e6ffe1603b8dc60fac7c33c18ae
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_6c104c1ee21349f637497ee67df4406e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 20, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_36b7a5f60efe6c5b9697037e93e4f22e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c104c1ee21349f637497ee67df4406e
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_5be2bab88f92de53c55c461d08d48f90(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d61a954d1b311e8d7e872982547de364(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5be2bab88f92de53c55c461d08d48f90
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_7f8fd291deb75a5a54e9a0884ffdf436(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cacc2c3109798831d958e89aa0a0a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f8fd291deb75a5a54e9a0884ffdf436
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b74c4a1cb203e04e804e9f293db5d709(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_640005980e98bc23152f5e80700fb688
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15ed3a5801da0d202c9d3f0b0cddb5b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09e65187b8cb4febec5e642b51e28075
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11767a7c55fa4d323bf35db09f010dbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5be2bab88f92de53c55c461d08d48f90
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fb81ecd7b424f011ba1360a1bd674ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f8fd291deb75a5a54e9a0884ffdf436
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5de2c5ec6469010c7eeef7101e36bb9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6cd227eaad1d7d867bb2b41cab8bda
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb456a38bd8ca94ab751f7deba383e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d288336a9f8efdd256e8e3d480df475
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89ee268719c2eb1274c3fe51b4f00562(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2349e6ffe1603b8dc60fac7c33c18ae
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_36b7a5f60efe6c5b9697037e93e4f22e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c104c1ee21349f637497ee67df4406e
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46e5353d1b15eae4459605671096e7ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db4ab4381ed71c213379c921f858216b
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_d23e959f956ad222bed904ffe13541f0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 32, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_afd241cdc80f2ee4133c303b12652fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d23e959f956ad222bed904ffe13541f0
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_b6014dfa4d8824a404c72e5f5b9eb3b9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 8, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f1257d18d1c98948b7298c44c4125df0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6014dfa4d8824a404c72e5f5b9eb3b9
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_14051a2e3c8dfee362b16792e8045462(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c2349e6ffe1603b8dc60fac7c33c18ae
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60d231b8e3217dc321676d46352340a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c104c1ee21349f637497ee67df4406e
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_5de2c5ec6469010c7eeef7101e36bb9f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6cd227eaad1d7d867bb2b41cab8bda
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb456a38bd8ca94ab751f7deba383e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d288336a9f8efdd256e8e3d480df475
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2230903b0625f7c962401873712c24a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5be2bab88f92de53c55c461d08d48f90
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fb81ecd7b424f011ba1360a1bd674ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f8fd291deb75a5a54e9a0884ffdf436
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbfe9cea52c8707da1396100a7dfb538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09e65187b8cb4febec5e642b51e28075
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d1e6d2653efc4dbfee68dc343b96169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d23e959f956ad222bed904ffe13541f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb1e02dba46a7dac725223ae80d51cde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6014dfa4d8824a404c72e5f5b9eb3b9
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e472290bea3b005342ad14ca2df8736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5be2bab88f92de53c55c461d08d48f90
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cacc2c3109798831d958e89aa0a0a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f8fd291deb75a5a54e9a0884ffdf436
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cacc2c3109798831d958e89aa0a0a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f8fd291deb75a5a54e9a0884ffdf436
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ccdf031400afd8ef442a0998cc594c49(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22c2c0b908f3c98bbc02b4ba02fdc834
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_46e5353d1b15eae4459605671096e7ce(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db4ab4381ed71c213379c921f858216b
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2230903b0625f7c962401873712c24a0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5be2bab88f92de53c55c461d08d48f90
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fb81ecd7b424f011ba1360a1bd674ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f8fd291deb75a5a54e9a0884ffdf436
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c78599a1a5f94f7120ebebcb0bd6befd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d288336a9f8efdd256e8e3d480df475
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b89c21ddf4301e10abdf3729da1e6662(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_640005980e98bc23152f5e80700fb688
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_15ed3a5801da0d202c9d3f0b0cddb5b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09e65187b8cb4febec5e642b51e28075
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbfe9cea52c8707da1396100a7dfb538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09e65187b8cb4febec5e642b51e28075
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_270efa673579a8bd1c28cab1110b4c4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85343e1f2f9bbd0c9e3f706c66e0c39c
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5a8650d1a5862ff31fa8d9158dbbdde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59a69d4a5cc207d2b595686163be37d5
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_603ba589f9660a4b458d1af77d2219b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6cd227eaad1d7d867bb2b41cab8bda
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb456a38bd8ca94ab751f7deba383e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d288336a9f8efdd256e8e3d480df475
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_270efa673579a8bd1c28cab1110b4c4f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85343e1f2f9bbd0c9e3f706c66e0c39c
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a5a8650d1a5862ff31fa8d9158dbbdde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59a69d4a5cc207d2b595686163be37d5
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_603ba589f9660a4b458d1af77d2219b5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fd6cd227eaad1d7d867bb2b41cab8bda
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb456a38bd8ca94ab751f7deba383e58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6d288336a9f8efdd256e8e3d480df475
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7e472290bea3b005342ad14ca2df8736(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5be2bab88f92de53c55c461d08d48f90
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_2cacc2c3109798831d958e89aa0a0a60(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f8fd291deb75a5a54e9a0884ffdf436
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_60d231b8e3217dc321676d46352340a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c104c1ee21349f637497ee67df4406e
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7d1e6d2653efc4dbfee68dc343b96169(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d23e959f956ad222bed904ffe13541f0
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_cb1e02dba46a7dac725223ae80d51cde(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6014dfa4d8824a404c72e5f5b9eb3b9
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e458cb1b8e035843309e0ec65dbb3312(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_85343e1f2f9bbd0c9e3f706c66e0c39c
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fa289e8152778a0115a3c423d1e2256b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59a69d4a5cc207d2b595686163be37d5
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_afd241cdc80f2ee4133c303b12652fdd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d23e959f956ad222bed904ffe13541f0
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1257d18d1c98948b7298c44c4125df0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b6014dfa4d8824a404c72e5f5b9eb3b9
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fb8cd5558a04d80f69dbcbfe6c5545dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_640005980e98bc23152f5e80700fb688
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbfe9cea52c8707da1396100a7dfb538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09e65187b8cb4febec5e642b51e28075
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4725d05d7c515c1a2b48e6307e1d38ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_640005980e98bc23152f5e80700fb688
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_fbfe9cea52c8707da1396100a7dfb538(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_09e65187b8cb4febec5e642b51e28075
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_11767a7c55fa4d323bf35db09f010dbb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5be2bab88f92de53c55c461d08d48f90
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7fb81ecd7b424f011ba1360a1bd674ea(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7f8fd291deb75a5a54e9a0884ffdf436
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_47c04b649d4460274c08e877f637b02e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_22c2c0b908f3c98bbc02b4ba02fdc834
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d5bca7f32e8c4144f786a503dad1d61a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_db4ab4381ed71c213379c921f858216b
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            return paddle._C_ops.swish(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_184fea888c97096daf3ee326cd0abb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a58125c9460903f5aae301dc324689b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bfe54a76cdaa1c76d4e5b622f34f50f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_083e2bbc3104a693147782dc8315f378(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_184fea888c97096daf3ee326cd0abb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92c3649a974e34ff5a08e123810c729a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bfe54a76cdaa1c76d4e5b622f34f50f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0917f711b52238c91c659a6c0aaea4f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d44824e19967fd0f441b0c501075428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b9078c9af98473124caa132ce269bf40(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_184fea888c97096daf3ee326cd0abb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1a7fc40adb62fa47179d05409ee4d64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a17f5a7ae0257e5b245dd46b46d29123(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e63cfe28fcccbb00e0a43c9854970392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b9704538d9ef91f26215c10638b2b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_598d9b264e5c389ba2f89fd5cff495bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93c83a6440cf7865f684a0014af4a0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_92c3649a974e34ff5a08e123810c729a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bfe54a76cdaa1c76d4e5b622f34f50f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c96bf08ee69c944103105ea53df1904(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51b0880b91e77de96a0434b03506729d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1af963a6d89e24960da34bfaa9d417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a7d9def355f0e98389534044d28348(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_e63cfe28fcccbb00e0a43c9854970392(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7b9704538d9ef91f26215c10638b2b3d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b10170269d7fab45d229cfdd5f7096c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc75757d8885a8d07249073a31060cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_575884d191600b25cf40bc5edc3a1b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c0e7131d683ccec8de625f379264ac2d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 480, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01228010d6a0f4404c6e0caaecc856ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ce1af963a6d89e24960da34bfaa9d417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a7d9def355f0e98389534044d28348(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d338fd7b5dfebb4976d95086de1ffee8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51b0880b91e77de96a0434b03506729d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_993807ee190d82ae7323c251463caef6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f3b4ca9c0ab8c42bc67078ad6ef9a0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29f36605fcc44db94de039118ed211f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5bbdac89e0ca48ee7e3449e425a4ea5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93c83a6440cf7865f684a0014af4a0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93c83a6440cf7865f684a0014af4a0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_05985906d5f2ac79578346ce15ba7a3b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_8b10170269d7fab45d229cfdd5f7096c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_d338fd7b5dfebb4976d95086de1ffee8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51b0880b91e77de96a0434b03506729d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_184fea888c97096daf3ee326cd0abb8d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7a58125c9460903f5aae301dc324689b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4bfe54a76cdaa1c76d4e5b622f34f50f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_993807ee190d82ae7323c251463caef6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f54119347473a31d1b65d848b31569b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44ffae670baca36c67a9401ccd46da84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_430c385ee332c443560321b4ffcd4cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a7d9def355f0e98389534044d28348(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f54119347473a31d1b65d848b31569b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_44ffae670baca36c67a9401ccd46da84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_430c385ee332c443560321b4ffcd4cb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 672, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_45a7d9def355f0e98389534044d28348(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 28, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5bbdac89e0ca48ee7e3449e425a4ea5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 144, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a93c83a6440cf7865f684a0014af4a0b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_01228010d6a0f4404c6e0caaecc856ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 20, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0f3b4ca9c0ab8c42bc67078ad6ef9a0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_29f36605fcc44db94de039118ed211f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f1a7fc40adb62fa47179d05409ee4d64(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 96, 56, 56], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a17f5a7ae0257e5b245dd46b46d29123(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 4, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_dc75757d8885a8d07249073a31060cdb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 32, 112, 112], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_575884d191600b25cf40bc5edc3a1b69(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 8, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_a9602b549f4840ff6ab989088a239f29(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 14, 14], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_993807ee190d82ae7323c251463caef6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_89788c4df08620322ae4395478ac1fc6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 240, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_993807ee190d82ae7323c251463caef6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([11, 10, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0c96bf08ee69c944103105ea53df1904(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 144, 28, 28], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_51b0880b91e77de96a0434b03506729d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 6, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0917f711b52238c91c659a6c0aaea4f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 1152, 7, 7], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1d44824e19967fd0f441b0c501075428(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_816b0354f784cb8e3ad8cae71cc3bcee
        def get_inputs(self):
            return [
                paddle.uniform([43, 48, 1, 1], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()