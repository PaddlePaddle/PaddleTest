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
    class PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f32c930540104b1fbc115353f570d615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.41601404547691345]], [[0.17238618433475494]], [[0.08263786882162094]], [[0.13775232434272766]], [[0.37536704540252686]], [[0.2121986299753189]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_043125baf4b9e087f6861e7b11a2f38c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2972797751426697]], [[0.225433811545372]], [[0.3709498941898346]], [[0.04370572045445442]], [[0.3836890459060669]], [[0.4882890582084656]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_c1ca620bb5b6c493634bfa020b699dcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4aedc441f3e92f6aac4d60e3446d2558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_deb157d01fc311dd9fcf591399e137e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_735109d0bdc5f86dd3dbdd9b47a2496f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.15741729736328125]], [[0.35227862000465393]], [[0.4430646300315857]], [[0.45062822103500366]], [[0.056922364979982376]], [[0.15342780947685242]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_8f797aff3b6ff36d6ee0366ae8186417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06528989225625992]], [[0.41746431589126587]], [[0.12354917079210281]], [[0.3680232763290405]], [[0.05339458957314491]], [[0.011256362311542034]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7a355a4249fcd8b6010ef57c7638f01c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.417911559343338], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0771025cfaf965d4b15ff9735414c67c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3602656126022339], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1103f8f0b4142faba4b969e5ae8c78c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f52dd9cd92fb5b7b3cfe8f94111ec9d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.01616477221250534], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bfc533fceb8b31b525bfc39d5a701331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4549250304698944], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_52381c2634e364b52f1f4d266fa9b622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.38430914282798767], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_94f958a5a1e6921158b7ec266dc2f921(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.271852046251297], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5dfca3bff8bf0e3a1f677edff9e9521e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.13239459693431854], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4aa96dd6a26d22c7bfb5b29f26320495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02385985292494297], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1d7d43a0d62feb48a35d5192a240645e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.25204846262931824], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9b582a29b1098499b845d0d9410d6e5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3354078531265259], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b3456fab3f772fa4159a1c7b0828fe20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3546801507472992], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_c49f69e2c7299a0f35082bdafea85f4b(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None, None, None, None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7897c487d8cec111a576ae492c64915b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c49f69e2c7299a0f35082bdafea85f4b
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6093430d863f35a198c9da6300ba7799(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0425312cd60bcc4007d968214a203231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[6, 1, 1], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_ce1b6df01410c607c312176969a8950f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.41601404547691345]], [[0.17238618433475494]], [[0.08263786882162094]], [[0.13775232434272766]], [[0.37536704540252686]], [[0.2121986299753189]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_9fdeaa8f5a1e43580098154493708766(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2972797751426697]], [[0.225433811545372]], [[0.3709498941898346]], [[0.04370572045445442]], [[0.3836890459060669]], [[0.4882890582084656]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_a9f5f937ac88672692fecdc9ac9f0d77(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 12096, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_d82123876f8a7808252336ed78ec57dd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a9f5f937ac88672692fecdc9ac9f0d77
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_e96f8c17217509f96fb67d5c2b682a6a(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6, 21824], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_bd55103ac374c7b6a472e64ed78c0cc2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_e96f8c17217509f96fb67d5c2b682a6a
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_67df43b3a57fc65c9e732397082fd18e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 5376, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_f401d0d49ac6a7cc2ef4720e1f709fb8(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_67df43b3a57fc65c9e732397082fd18e
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4688decd6b80cc56ec1f2b769d7f05b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.15741729736328125]], [[0.35227862000465393]], [[0.4430646300315857]], [[0.45062822103500366]], [[0.056922364979982376]], [[0.15342780947685242]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_2615cf2bcb3cdb9287690a3de4f22fb0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06528989225625992]], [[0.41746431589126587]], [[0.12354917079210281]], [[0.3680232763290405]], [[0.05339458957314491]], [[0.011256362311542034]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_7a355a4249fcd8b6010ef57c7638f01c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.417911559343338], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0771025cfaf965d4b15ff9735414c67c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3602656126022339], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_efde19a30c7a4ebd010a3fca7907c2da(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 8400, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_8fb83bb245ccddc95442f78fa404207d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_efde19a30c7a4ebd010a3fca7907c2da
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f52dd9cd92fb5b7b3cfe8f94111ec9d7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.01616477221250534], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bfc533fceb8b31b525bfc39d5a701331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4549250304698944], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_52381c2634e364b52f1f4d266fa9b622(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.38430914282798767], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_94f958a5a1e6921158b7ec266dc2f921(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.271852046251297], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_5dfca3bff8bf0e3a1f677edff9e9521e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.13239459693431854], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4aa96dd6a26d22c7bfb5b29f26320495(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02385985292494297], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1d7d43a0d62feb48a35d5192a240645e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.25204846262931824], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9b582a29b1098499b845d0d9410d6e5f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3354078531265259], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b3456fab3f772fa4159a1c7b0828fe20(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3546801507472992], dtype='float32').reshape([1]),
            ]


    
    class PrimitiveOp_d41f8bab56c9ee4c9166af8e00ef8a7e(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[2, 1, 960, 960], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_7b3fabc33eee123f38cf50157ef15235(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_d41f8bab56c9ee4c9166af8e00ef8a7e
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_9815ec78f5385b8f6d3cad3db4eebfff(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6069, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5111a7ee6f3d19b3c5ab58c55ed226ae(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_9815ec78f5385b8f6d3cad3db4eebfff
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    
    class PrimitiveOp_13215c4a800bc3b686f789aaa1afd894(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[1, 6804, 2], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_e67b4d580d28e8f017728686ca597696(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_13215c4a800bc3b686f789aaa1afd894
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_f32c930540104b1fbc115353f570d615(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.41601404547691345]], [[0.17238618433475494]], [[0.08263786882162094]], [[0.13775232434272766]], [[0.37536704540252686]], [[0.2121986299753189]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_043125baf4b9e087f6861e7b11a2f38c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2972797751426697]], [[0.225433811545372]], [[0.3709498941898346]], [[0.04370572045445442]], [[0.3836890459060669]], [[0.4882890582084656]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_c1ca620bb5b6c493634bfa020b699dcd(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 12096, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_4aedc441f3e92f6aac4d60e3446d2558(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6, 21824], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_deb157d01fc311dd9fcf591399e137e7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 5376, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_735109d0bdc5f86dd3dbdd9b47a2496f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.15741729736328125]], [[0.35227862000465393]], [[0.4430646300315857]], [[0.45062822103500366]], [[0.056922364979982376]], [[0.15342780947685242]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_8f797aff3b6ff36d6ee0366ae8186417(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06528989225625992]], [[0.41746431589126587]], [[0.12354917079210281]], [[0.3680232763290405]], [[0.05339458957314491]], [[0.011256362311542034]]], dtype='float32').reshape([6, 1, 1]),
            ]


    
    class PrimitiveOp_2333867b78301992e5f52c862ef62ac9(InstanceTrait, paddle.nn.Layer):
        
        def __init__(self):
            super().__init__()

        def forward(self, input_0):
            return paddle._C_ops.exp(input_0)

        def get_input_spec(self):
            return [
                paddle.static.InputSpec(shape=[None], dtype='float32'),
            ]
            
        instance_ = None
        static_instance_with_cinn_ = None
        static_instance_without_cinn_ = None


    class TestPrimitiveOp_5c8f1fdb82d5a85f3c1bf3035f61621d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.417911559343338], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1e168d16547096666a82909d51669c1c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3602656126022339], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1103f8f0b4142faba4b969e5ae8c78c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_ec72fdc17cc38029ec649c7acf81205a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.01616477221250534], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_af498c306105d5695da8167af705da0c(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4549250304698944], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_e257ddb36736320bce777c4bacb75486(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.38430914282798767], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_dc5d15904a1494d823c8ec9a066652c0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.271852046251297], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_0717beecfa300847c4c29722c1d9ae91(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.13239459693431854], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7d91cd9930316baeaf5824a6758c26f1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02385985292494297], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ab9cf92020dc189b216c3d55a4ee036f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.25204846262931824], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_80932738bf986c04120372bbe1cf9146(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3354078531265259], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_66f2874ef9a94d0e5f0031d64b9c2a9d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3546801507472992], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_7897c487d8cec111a576ae492c64915b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_c49f69e2c7299a0f35082bdafea85f4b
        def get_inputs(self):
            return [
                paddle.uniform([2, 1, 960, 960], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_6093430d863f35a198c9da6300ba7799(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6069, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_0425312cd60bcc4007d968214a203231(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 6804, 2], dtype='float32', min=0, max=0.5),
            ]


    

if __name__ == '__main__':
    unittest.main()