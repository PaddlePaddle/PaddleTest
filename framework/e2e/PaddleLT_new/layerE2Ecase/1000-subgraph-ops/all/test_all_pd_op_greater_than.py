import os
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
    class PrimitiveOp_4e52b078e2c114f3836b3888c4648d0f(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c0e5be2bd4d33c0b9a384c55f6d8b298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e52b078e2c114f3836b3888c4648d0f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.0, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_a865333ac980c6731742ef8ef422de13(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 500, 128], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_adceecdf7d3f38d78fbdcad968502fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a865333ac980c6731742ef8ef422de13
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_adceecdf7d3f38d78fbdcad968502fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a865333ac980c6731742ef8ef422de13
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_148b315ac12fc4d2ecb452e1bc4939a4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_99400664ec6c0d9285bf836ab6e6006b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_148b315ac12fc4d2ecb452e1bc4939a4
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 6], dtype='int32').reshape([2]),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_0f5176b793bd2646a2cb255b6ceac5df(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c41750c019d7d2c48ba04401248ca44f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f5176b793bd2646a2cb255b6ceac5df
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.18129011988639832, 0.4002115726470947, 0.3979283273220062, 0.3470446765422821, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.4002115726470947, 0.4761844575405121, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f2ad52899479fbfef943ac76df16895b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f5176b793bd2646a2cb255b6ceac5df
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.35412517189979553, 0.2729951739311218, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.43007832765579224, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_eff732dee792efbd4f03714dc811f07a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_148b315ac12fc4d2ecb452e1bc4939a4
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 9], dtype='int32').reshape([2]),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_38deed57335f4edcbcd3095e30ff76f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e52b078e2c114f3836b3888c4648d0f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.0, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_3b1f14c5e0937b9c4fb141e90a3dff97(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_97fab01b9d69105d7240bf21e15ee032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b1f14c5e0937b9c4fb141e90a3dff97
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c27af15117e1900c4bd166998d5d10b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b1f14c5e0937b9c4fb141e90a3dff97
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e9890d040485939c7c7f29c9645b0d28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e52b078e2c114f3836b3888c4648d0f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_adceecdf7d3f38d78fbdcad968502fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a865333ac980c6731742ef8ef422de13
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e04ab0ad339f58e7d33758cb8a42d0cd(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 2100], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5048ec9d233d6d8a81ac3e70e0740cee(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e04ab0ad339f58e7d33758cb8a42d0cd
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_adceecdf7d3f38d78fbdcad968502fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a865333ac980c6731742ef8ef422de13
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_adceecdf7d3f38d78fbdcad968502fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a865333ac980c6731742ef8ef422de13
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_e989f0573f634b43088b80055abc1a3a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_24bb03f4d984221fe9e693231eed1874(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e989f0573f634b43088b80055abc1a3a
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 6], dtype='int32').reshape([2]),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_9fcbcc3714797685e843dbd33618d826(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6], dtype='float32'),
                paddle.static.InputSpec(shape=[6], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_0f59fc5901d6c0b5ac8340ddd4fed0d8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fcbcc3714797685e843dbd33618d826
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.18129011988639832, 0.4002115726470947, 0.3979283273220062, 0.3470446765422821, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.4002115726470947, 0.4761844575405121, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_4a10a69c0ec25360b9210e6a7b08f086(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fcbcc3714797685e843dbd33618d826
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.35412517189979553, 0.2729951739311218, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.43007832765579224, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_6c84a59ef5897a1d7b6e2d7bdac9f4a5(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e989f0573f634b43088b80055abc1a3a
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 9], dtype='int32').reshape([2]),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_9fc80f768214ea3220bcd2adea706307(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 3549], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a14225bf87de83817cda258d9b7d80a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9fc80f768214ea3220bcd2adea706307
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_97fab01b9d69105d7240bf21e15ee032(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b1f14c5e0937b9c4fb141e90a3dff97
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c27af15117e1900c4bd166998d5d10b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_3b1f14c5e0937b9c4fb141e90a3dff97
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    
    class PrimitiveOp_a2573803208250c514b9f59a9630c9e2(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 4116], dtype='float32'),
                paddle.static.InputSpec(shape=[], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_c1cd9d102cc28e7ae6fc4e0d9e57ead6(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a2573803208250c514b9f59a9630c9e2
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_adceecdf7d3f38d78fbdcad968502fa1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a865333ac980c6731742ef8ef422de13
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c0e5be2bd4d33c0b9a384c55f6d8b298(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e52b078e2c114f3836b3888c4648d0f
        def get_inputs(self):
            return [
                paddle.uniform([1, 2100], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.0, dtype='float32').reshape([]),
            ]


    
    class PrimitiveOp_1ca0f8cbffc3fce55fd52caa06a9ffca(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0, input_1):
            return input_0 > input_1

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='int32'),
                paddle.static.InputSpec(shape=[], dtype='int32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8914de12d6d6b50fbc17c26905018207(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ca0f8cbffc3fce55fd52caa06a9ffca
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_8914de12d6d6b50fbc17c26905018207(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ca0f8cbffc3fce55fd52caa06a9ffca
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_99400664ec6c0d9285bf836ab6e6006b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_148b315ac12fc4d2ecb452e1bc4939a4
        def get_inputs(self):
            return [
                paddle.to_tensor([2, 6], dtype='int32').reshape([2]),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_c41750c019d7d2c48ba04401248ca44f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f5176b793bd2646a2cb255b6ceac5df
        def get_inputs(self):
            return [
                paddle.to_tensor([0.32048559188842773, 0.18129011988639832, 0.4002115726470947, 0.3979283273220062, 0.3470446765422821, 0.17776672542095184], dtype='float32').reshape([6]),
                paddle.to_tensor([0.4498527944087982, 0.4890660345554352, 0.4002115726470947, 0.4761844575405121, 0.44280150532722473, 0.17776672542095184], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_f2ad52899479fbfef943ac76df16895b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_0f5176b793bd2646a2cb255b6ceac5df
        def get_inputs(self):
            return [
                paddle.to_tensor([0.3242815434932709, 0.39286479353904724, 0.35412517189979553, 0.2729951739311218, 0.3105509877204895, 0.32766973972320557], dtype='float32').reshape([6]),
                paddle.to_tensor([0.1643696129322052, 0.4882410168647766, 0.43661245703697205, 0.43007832765579224, 0.49191561341285706, 0.33490511775016785], dtype='float32').reshape([6]),
            ]


    class TestPrimitiveOp_eff732dee792efbd4f03714dc811f07a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_148b315ac12fc4d2ecb452e1bc4939a4
        def get_inputs(self):
            return [
                paddle.to_tensor([6, 9], dtype='int32').reshape([2]),
                paddle.to_tensor(-1, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_38deed57335f4edcbcd3095e30ff76f2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e52b078e2c114f3836b3888c4648d0f
        def get_inputs(self):
            return [
                paddle.uniform([1, 3549], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_49065750ac81d4c0ca12c1e8c2a5c62e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_148b315ac12fc4d2ecb452e1bc4939a4
        def get_inputs(self):
            return [
                paddle.to_tensor([7], dtype='int32').reshape([1]),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_babd4f2b1fb3454f5cc58dea1826893b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_148b315ac12fc4d2ecb452e1bc4939a4
        def get_inputs(self):
            return [
                paddle.to_tensor([3], dtype='int32').reshape([1]),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    class TestPrimitiveOp_e9890d040485939c7c7f29c9645b0d28(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_4e52b078e2c114f3836b3888c4648d0f
        def get_inputs(self):
            return [
                paddle.uniform([1, 4116], dtype='float32', min=0, max=0.5),
                paddle.to_tensor(0.0, dtype='float32').reshape([]),
            ]


    class TestPrimitiveOp_8914de12d6d6b50fbc17c26905018207(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_1ca0f8cbffc3fce55fd52caa06a9ffca
        def get_inputs(self):
            return [
                paddle.randint(low=0, high=3, shape=[1, 500, 128], dtype='int32'),
                paddle.to_tensor(0, dtype='int32').reshape([]),
            ]


    

if __name__ == '__main__':
    unittest.main()