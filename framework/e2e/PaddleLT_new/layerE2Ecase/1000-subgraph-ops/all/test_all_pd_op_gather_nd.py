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
    class PrimitiveOp_d9fc4c8470634c3c1674575ddcf14aa6(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 80, 28, 28], dtype='float32'),
                paddle.static.InputSpec(shape=[None, 4], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_463b1fe985267e4d3e7a213df32c592d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9fc4c8470634c3c1674575ddcf14aa6
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3136, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_fdc47f2c9017e6653445c84776939bb3(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, None, 128], dtype='float32'),
                paddle.static.InputSpec(shape=[1, 500, 2], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_578a6ea053db5fc9f9285d514259ab77(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdc47f2c9017e6653445c84776939bb3
        def get_inputs(self):
            return [
                paddle.uniform([1, 41344, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    class TestPrimitiveOp_4f0a4eda6cab3198d7454309d7702671(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdc47f2c9017e6653445c84776939bb3
        def get_inputs(self):
            return [
                paddle.uniform([1, 25920, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    class TestPrimitiveOp_e896d0a0a418a77d71e3b804988e5063(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9fc4c8470634c3c1674575ddcf14aa6
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2352, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_4b7310ab21c25f2600305252cc2532f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9fc4c8470634c3c1674575ddcf14aa6
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4704, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_3aed4f7c21d4f9baf84b16db64d94ac4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d9fc4c8470634c3c1674575ddcf14aa6
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1568, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_f319fd1c291e9b283a3ac7f594fb3059(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_fdc47f2c9017e6653445c84776939bb3
        def get_inputs(self):
            return [
                paddle.uniform([1, 11520, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    
    class PrimitiveOp_608cbe566d9efa8d311fb613760ae4fd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_91a2073129d14cdb9ebb393da692d0ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_608cbe566d9efa8d311fb613760ae4fd
        def get_inputs(self):
            return [
                paddle.uniform([4, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[3136, 4], dtype='int64'),
            ]


    
    class PrimitiveOp_7dd6845894b72089fde4247ee591d7d1(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0, arg_1):
            input_0 = arg_0
            input_1 = arg_1
            return paddle.gather_nd(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[None, None, None], dtype='int64'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_13d91a4c9b7ca931038c6a550842164b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dd6845894b72089fde4247ee591d7d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 41344, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    class TestPrimitiveOp_af525d4e1a775df53102c4ee0da2e55f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dd6845894b72089fde4247ee591d7d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 25920, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    class TestPrimitiveOp_6ee0583cd7a232842c17f919cde243ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_608cbe566d9efa8d311fb613760ae4fd
        def get_inputs(self):
            return [
                paddle.uniform([3, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[2352, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_47a70e192ef6214066f744e54d1b18a2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_608cbe566d9efa8d311fb613760ae4fd
        def get_inputs(self):
            return [
                paddle.uniform([6, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[4704, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_a37f15c3b006776551b6354de889d8b8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_608cbe566d9efa8d311fb613760ae4fd
        def get_inputs(self):
            return [
                paddle.uniform([2, 80, 28, 28], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1568, 4], dtype='int64'),
            ]


    class TestPrimitiveOp_915c2f5af472e9e8387f146881a71440(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7dd6845894b72089fde4247ee591d7d1
        def get_inputs(self):
            return [
                paddle.uniform([1, 11520, 128], dtype='float32', min=0, max=0.5),
                paddle.randint(low=0, high=3, shape=[1, 500, 2], dtype='int64'),
            ]


    

if __name__ == '__main__':
    unittest.main()