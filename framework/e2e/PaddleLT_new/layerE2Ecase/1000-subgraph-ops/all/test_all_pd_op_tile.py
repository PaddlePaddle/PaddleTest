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
    class PrimitiveOp_a172b4bc9b2f52ccd4d260304622714b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 300, 256], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_6186e34914670601ca7b7a6ada7b2dfb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a172b4bc9b2f52ccd4d260304622714b
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_02d897f5c8710f96eed924362787fb38(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8da5b92b7d0cdcca6eb3886f15f88afa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_46d4718b0cf468900bc75fb57e48f2cf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_4b2414f967ad35b650846f35462dfab2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_ec7d0c7360d743c8294890cbcc140905(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_8da5b92b7d0cdcca6eb3886f15f88afa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_31b53a861719ff017b53354bd2961131(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 76]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, 1], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_9ecc885574cec112dfb13e63c735097f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_31b53a861719ff017b53354bd2961131
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_bbe70448c99edc9646fb9aa4765d204d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_8ec0c7c1c3d3ac625670cbd6ab3a1e4a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_7920907a7bacbbc7e52119e76340af5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_51d256ff131ce0816ff75f09b9beb4d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_e3ef62e5d36e1903283c5bdd839e89bb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_07af67c1f56ea3650dbcef0f5f6d643a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_f8ae2091607e37b6acc0e52d7127d8d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_33a3cdafa1f21de4957f3e08d43dabee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_7b616828e25ebe3d128c87bf911ec917(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 100, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_2cd9ef26d2781e7b4d9e7bdbe51738f5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_7b616828e25ebe3d128c87bf911ec917
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.37339529395103455, 0.30681005120277405, 0.14150168001651764, 0.0019667954184114933]]], dtype='float32').reshape([1, 1, 4]),
            ]


    
    class PrimitiveOp_4bca4bad98796e406f81cdaa65aa21c0(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 300, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 1, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8749183b51a4d72f5891994a5d6f64f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4bca4bad98796e406f81cdaa65aa21c0
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.015704719349741936, 0.1316700279712677, 0.0756133571267128, 0.4947950839996338]]], dtype='float32').reshape([1, 1, 4]),
            ]


    class TestPrimitiveOp_8b10f197d30a92eb57aab406957acbcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_ec41aaa5abdbc23ec76d162df20cfa10(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_e538570a05d040ce6a836c2878640bca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_317fa0c209212a63f1eda7ce704cb8aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_5baef5ca93d7cbfcab95a4ad595a0fb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_0ec9565048b38b8a7a85ef482b83bb87(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_6c3e34f1281206d1d7f4c105f467dbfd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 512]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, 512, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d4f32f55768a8bd51ef6dc80b79c0bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c3e34f1281206d1d7f4c105f467dbfd
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_7920907a7bacbbc7e52119e76340af5a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_51d256ff131ce0816ff75f09b9beb4d4(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_d4f32f55768a8bd51ef6dc80b79c0bfd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_6c3e34f1281206d1d7f4c105f467dbfd
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_668d6412c7a90001d1e6bee08f7d88fe(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_02d897f5c8710f96eed924362787fb38
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_836902e50475c3044a92a98a32fcffeb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_f77751a7ae2088105cffba4bdf8b6254
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_c6061fd6585a0fa95488e3ed078a9746(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_84b7c5c5ae70729d5f78c626cba9041f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c6061fd6585a0fa95488e3ed078a9746
        def get_inputs(self):
            return [
                paddle.uniform([1, 300, 256], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 4]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_952718568df090ac1774ccac8991938e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_e358c7d94a6f916b4ffa05a887532949(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 68]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ddf50879df66e8757cf126ac4fa0b4c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_c86ad73ec3c7d8ca08baf18baa133e52(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_b231d819466b63b09538a6ad08d79956(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 11109, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_952718568df090ac1774ccac8991938e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_9f4f5eba65feeb00c4257a4a62800946(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 76]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_cb8cd8673baa8e4905d3f6c4daab1724(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9f4f5eba65feeb00c4257a4a62800946
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3549, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_b5f1b3d03f975b68005967a61de39116(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_a5579e9e12822b523bb4ea15a98815a6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 3024, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_0fe83cca7407f31f7510c31988e23cb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_7fc577e6293cd87455b1f51a883ca1a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_646914e620d4b381026b32fcc0fc2eb5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_c8b78284c8560d06752b2844cb0d61ca(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 9261, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_88e6410dd3e7ef021edeab96ca510f44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_210bf080f2db03802b14402214da047c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 2100, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_5fbd059ecbbe884cc90a4d7ea2704785(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 100, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_75f38a7d2a12f7e64fc24215546c1e81(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_5fbd059ecbbe884cc90a4d7ea2704785
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.37339529395103455, 0.30681005120277405, 0.14150168001651764, 0.0019667954184114933]]], dtype='float32').reshape([1, 1, 4]),
            ]


    
    class PrimitiveOp_d00970d548826c11a30d32bc1af44517(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 300, 1]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0b5aa48ab3f34b6aac4fb39ec7cb080f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d00970d548826c11a30d32bc1af44517
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.015704719349741936, 0.1316700279712677, 0.0756133571267128, 0.4947950839996338]]], dtype='float32').reshape([1, 1, 4]),
            ]


    class TestPrimitiveOp_3642fbdd5429941887ebb1b39b352dd7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_94e14f732f5367c9bb2d721b11faef55(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4725, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_f4805250878a641c982dfc3dd6d2b20b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_67181a8c9fe5727c3f68ec8af27686a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 6069, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_e7301ad0f969614db6a8bb6cd2f3b146(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_44384fc126e74574647eded38b37ef2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 7581, 1], dtype='int32'),
            ]


    
    class PrimitiveOp_b1384d91c23a3a4fe7f240deb3b2ef20(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, arg_0):
            input_0 = arg_0
            input_1 = [1, 1, 512]
            return paddle._C_ops.tile(input_0, input_1)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ffd325a229ab077996831eb078f69ec0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1384d91c23a3a4fe7f240deb3b2ef20
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0fe83cca7407f31f7510c31988e23cb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_7fc577e6293cd87455b1f51a883ca1a1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 4116, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_ffd325a229ab077996831eb078f69ec0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_b1384d91c23a3a4fe7f240deb3b2ef20
        def get_inputs(self):
            return [
                paddle.uniform([1, 512, 1], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_b611d3bb8f1a6fa80193bc0b9433bb7f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c186ca4f0f69b4d86344ed77b8adb341
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            ]


    class TestPrimitiveOp_319da243a05dcebac44e755a1926810c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e358c7d94a6f916b4ffa05a887532949
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 8400, 1], dtype='int32'),
            ]


    

if __name__ == '__main__':
    unittest.main()