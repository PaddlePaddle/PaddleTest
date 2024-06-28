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


    class TestPrimitiveOp_48f36a39fd5f0cbe9658908aee4d7ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.46238023042678833]], [[0.4869759678840637]], [[0.25384998321533203]], [[0.3568928837776184]], [[0.3896326422691345]], [[0.24136339128017426]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_a5f72d72cae765fae87f931aedbe3174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.21430465579032898]], [[0.27133625745773315]], [[0.3204556703567505]], [[0.42410537600517273]], [[0.06479871273040771]], [[0.26669731736183167]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_063f633fe7efcdeb3afed02b26b891ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2877563238143921]], [[0.4828806519508362]], [[0.40909233689308167]], [[0.19218365848064423]], [[0.35922595858573914]], [[0.4864318370819092]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_bde9fba84328812f8c310efba6c9e63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06651467829942703]], [[0.02692185342311859]], [[0.13614360988140106]], [[0.13175329566001892]], [[0.35920295119285583]], [[0.41328200697898865]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_61343f6721cf02290eb50ac0fdbf0f19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4983212947845459], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f5658de69fdd3e0f9c2bde7647b74c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.25297215580940247], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1103f8f0b4142faba4b969e5ae8c78c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_403c9c26b1335e45c8f58f4f6b83b0eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4190675914287567], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9f1c873991b92cabd963f63262c2d557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.28315821290016174], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d8efb0d269b89850038fdd3141ecd4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3482629954814911], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3fc9c4c93e7816c6cb5c3f35f0ac36ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.16106460988521576], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a5a0d65f2d315a4936f2f5bb8b65ab6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4513109028339386], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4f7d31a12efd991127372895ee7022f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.15913958847522736], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_96c937daa412bb0ebb84476493cdc352(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3949134349822998], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_36752de5c45b18ceb20f7c47d8d90331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02164243534207344], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ad77c73eae00f376ac9cc344c642faad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.026567328721284866], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_19ab7f781db4744f686abdce49ec7716(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.46238023042678833]], [[0.4869759678840637]], [[0.25384998321533203]], [[0.3568928837776184]], [[0.3896326422691345]], [[0.24136339128017426]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_0d9055bbe1f2d4b87624c4d3752bceff(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.21430465579032898]], [[0.27133625745773315]], [[0.3204556703567505]], [[0.42410537600517273]], [[0.06479871273040771]], [[0.26669731736183167]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_6340e7fe23db47ed02a76d6e751ac914(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2877563238143921]], [[0.4828806519508362]], [[0.40909233689308167]], [[0.19218365848064423]], [[0.35922595858573914]], [[0.4864318370819092]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_f7d056a5a2799e44604e687720ae1af0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06651467829942703]], [[0.02692185342311859]], [[0.13614360988140106]], [[0.13175329566001892]], [[0.35920295119285583]], [[0.41328200697898865]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_61343f6721cf02290eb50ac0fdbf0f19(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4983212947845459], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f5658de69fdd3e0f9c2bde7647b74c47(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.25297215580940247], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_403c9c26b1335e45c8f58f4f6b83b0eb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4190675914287567], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9f1c873991b92cabd963f63262c2d557(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.28315821290016174], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d8efb0d269b89850038fdd3141ecd4aa(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3482629954814911], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3fc9c4c93e7816c6cb5c3f35f0ac36ba(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.16106460988521576], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a5a0d65f2d315a4936f2f5bb8b65ab6a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4513109028339386], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_4f7d31a12efd991127372895ee7022f9(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.15913958847522736], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_96c937daa412bb0ebb84476493cdc352(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3949134349822998], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_36752de5c45b18ceb20f7c47d8d90331(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02164243534207344], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ad77c73eae00f376ac9cc344c642faad(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.026567328721284866], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_48f36a39fd5f0cbe9658908aee4d7ca1(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.46238023042678833]], [[0.4869759678840637]], [[0.25384998321533203]], [[0.3568928837776184]], [[0.3896326422691345]], [[0.24136339128017426]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_a5f72d72cae765fae87f931aedbe3174(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.21430465579032898]], [[0.27133625745773315]], [[0.3204556703567505]], [[0.42410537600517273]], [[0.06479871273040771]], [[0.26669731736183167]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_063f633fe7efcdeb3afed02b26b891ec(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2877563238143921]], [[0.4828806519508362]], [[0.40909233689308167]], [[0.19218365848064423]], [[0.35922595858573914]], [[0.4864318370819092]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_bde9fba84328812f8c310efba6c9e63e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.06651467829942703]], [[0.02692185342311859]], [[0.13614360988140106]], [[0.13175329566001892]], [[0.35920295119285583]], [[0.41328200697898865]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_d211ab1bb3421797eb537ea217f75402(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4983212947845459], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_b67bb108d9aec90847b1c172fcc3db32(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.25297215580940247], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1103f8f0b4142faba4b969e5ae8c78c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_443806e4183f6c8bc1d0dcbf02bbf7b3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4190675914287567], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_14a09d784fdfe6ed64aa0f7059d98f42(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.28315821290016174], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_2d6606872cc743843d6da7a829412a58(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3482629954814911], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ccfab883850ac30dff68883027a1e778(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.16106460988521576], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3be90ddecb5b52346e8d8b39ac2ce8a7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4513109028339386], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_fb13c5c741d8f0d6d617d7ff6b384631(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.15913958847522736], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_9efccbba77404990a7a5d261225249f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.3949134349822998], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_6c8eacb5b299684fff3e5a712d86a136(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.02164243534207344], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_100f0266be5239390b16bb9a7663d526(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.026567328721284866], dtype='float32').reshape([1]),
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