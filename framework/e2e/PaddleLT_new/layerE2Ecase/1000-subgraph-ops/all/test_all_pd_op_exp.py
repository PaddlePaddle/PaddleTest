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


    class TestPrimitiveOp_77caa8075f1aa051a7f1434ca58081cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.3182089924812317]], [[0.1517808586359024]], [[0.46998053789138794]], [[0.08913518488407135]], [[0.20686282217502594]], [[0.1967306286096573]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_d10d9c18d628f85c946771772f2f6861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.23603032529354095]], [[0.332725465297699]], [[0.37263891100883484]], [[0.0956811010837555]], [[0.4223902225494385]], [[0.41398653388023376]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_334cfd9bc0e4a48f18d03d944f897585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.18453159928321838]], [[0.12145353853702545]], [[0.3111409544944763]], [[0.3378380537033081]], [[0.3507905602455139]], [[0.22727002203464508]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_445ed434cc43d4f894bff7460f76a5f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2972039580345154]], [[0.47957664728164673]], [[0.4044768512248993]], [[0.19967120885849]], [[0.3705017566680908]], [[0.1407492607831955]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_8b24312f1eb05e7da1963cf137347866(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.28897014260292053], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c541d2a675da05ab63439479c9e7e820(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.029688622802495956], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1103f8f0b4142faba4b969e5ae8c78c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_1b3b87d1be16632f3045cd521b60ecb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.40666472911834717], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_457b0e94a84b91f0aa45ed033eda250b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.05183764174580574], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ba7a24f59558e739828dd3fb83c79555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.462546169757843], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a5fab745fc91948f5c680e1c732dce68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.374307245016098], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c27788bcfb6aa4525644453e70db632f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.14877818524837494], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d59aba34e8485cab99c188dba7d82321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2593044340610504], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_843393e29d9d3af4476210d5b728310b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4165513813495636], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bebbca7295024308d05a60c6f02ce2b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4602836072444916], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f2cff7b3537bcbd0c842293c8cc0bb84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4938599467277527], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_6f03faee9eda6cd32abd46cfd9a87a2e(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.3182089924812317]], [[0.1517808586359024]], [[0.46998053789138794]], [[0.08913518488407135]], [[0.20686282217502594]], [[0.1967306286096573]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_4b961582e39113864ed35e9f08f91b0a(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.23603032529354095]], [[0.332725465297699]], [[0.37263891100883484]], [[0.0956811010837555]], [[0.4223902225494385]], [[0.41398653388023376]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_a19e3fb766f367f3b8e2ce76fff9b673(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.18453159928321838]], [[0.12145353853702545]], [[0.3111409544944763]], [[0.3378380537033081]], [[0.3507905602455139]], [[0.22727002203464508]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_c68df7c8e3fc2bd790ba90357da6386d(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_98d872e65ef821c81c4eb54292fa87cf
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2972039580345154]], [[0.47957664728164673]], [[0.4044768512248993]], [[0.19967120885849]], [[0.3705017566680908]], [[0.1407492607831955]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_8b24312f1eb05e7da1963cf137347866(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.28897014260292053], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c541d2a675da05ab63439479c9e7e820(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.029688622802495956], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_1b3b87d1be16632f3045cd521b60ecb3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.40666472911834717], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_457b0e94a84b91f0aa45ed033eda250b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.05183764174580574], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ba7a24f59558e739828dd3fb83c79555(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.462546169757843], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_a5fab745fc91948f5c680e1c732dce68(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.374307245016098], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c27788bcfb6aa4525644453e70db632f(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.14877818524837494], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_d59aba34e8485cab99c188dba7d82321(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2593044340610504], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_843393e29d9d3af4476210d5b728310b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4165513813495636], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bebbca7295024308d05a60c6f02ce2b2(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4602836072444916], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_f2cff7b3537bcbd0c842293c8cc0bb84(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_59b8ef22daf1eac1da6512b0995e8323
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4938599467277527], dtype='float32').reshape([1]),
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


    class TestPrimitiveOp_77caa8075f1aa051a7f1434ca58081cb(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.3182089924812317]], [[0.1517808586359024]], [[0.46998053789138794]], [[0.08913518488407135]], [[0.20686282217502594]], [[0.1967306286096573]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_d10d9c18d628f85c946771772f2f6861(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.23603032529354095]], [[0.332725465297699]], [[0.37263891100883484]], [[0.0956811010837555]], [[0.4223902225494385]], [[0.41398653388023376]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_334cfd9bc0e4a48f18d03d944f897585(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.18453159928321838]], [[0.12145353853702545]], [[0.3111409544944763]], [[0.3378380537033081]], [[0.3507905602455139]], [[0.22727002203464508]]], dtype='float32').reshape([6, 1, 1]),
            ]


    class TestPrimitiveOp_445ed434cc43d4f894bff7460f76a5f7(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.to_tensor([[[0.2972039580345154]], [[0.47957664728164673]], [[0.4044768512248993]], [[0.19967120885849]], [[0.3705017566680908]], [[0.1407492607831955]]], dtype='float32').reshape([6, 1, 1]),
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


    class TestPrimitiveOp_0814fc7ba642f152946f77f304363c34(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.28897014260292053], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3895918fca7e7b403f00faee0bdbfa6b(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.029688622802495956], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_1103f8f0b4142faba4b969e5ae8c78c3(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_a487b5c598d6b618b407fc1c95f2caf4
        def get_inputs(self):
            return [
                paddle.uniform([1, 8400, 2], dtype='float32', min=0, max=0.5),
            ]


    class TestPrimitiveOp_c5839a19653b4a2ce7b8a0d1a63e4503(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.40666472911834717], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_baaa950f291c07c78267cda9e38a8c61(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.05183764174580574], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_ad51f42adc0e29df0464db1f72edc0e0(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.462546169757843], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_c9993da55b1b75baf3f373de19639a30(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.374307245016098], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_95ddecf22a70d5f17c85dbbb0bc88bbf(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.14877818524837494], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_8be5200f8cefdfa24693d2b76ddfcb53(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.2593044340610504], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_bd5d0a93cdccbd008c709271f737df44(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4165513813495636], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_3cf6a162975f0deddd15c5ee48a31103(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4602836072444916], dtype='float32').reshape([1]),
            ]


    class TestPrimitiveOp_abad9012da95c6e80846d946d3ea7667(CinnTestBase, unittest.TestCase):
        
        def get_test_class(self):
            return PrimitiveOp_2333867b78301992e5f52c862ef62ac9
        def get_inputs(self):
            return [
                paddle.to_tensor([-0.4938599467277527], dtype='float32').reshape([1]),
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